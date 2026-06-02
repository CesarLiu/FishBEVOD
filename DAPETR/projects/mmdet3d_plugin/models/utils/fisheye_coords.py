import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


class FisheyeCoordAug(nn.Module):
    """
        Compute MEI fisheye-aware per-pixel features and return them as extra channels.

        Features:
            - Unit ray directions (3ch) from MEI inverse projection (recommended)
            - Optional angle features derived from rays: cos/sin(phi), cos/sin(theta) (4ch)
            - Optional local area distortion channel (1ch): ||∂r/∂u × ∂r/∂v|| (log-standardized)

        Inputs:
            x:    (B*N, 3, H, W) image tensor (already augmented/padded)
            K:    (B*N, 3, 3) intrinsics for current H,W (after aug/pad). Supports skew K[0,1].
            dist: (B*N, 5) distortion params [k1, k2, p1, p2, xi]

        Returns:
            extra_feats: (B*N, F, H, W) where F depends on enabled features
    """

    def __init__(self, add_rays: bool = True, add_angles: bool = False, add_radial: bool = False,
                                 iters: int = 20, eps: float = 1e-8):
        super().__init__()
        self.add_rays = add_rays
        self.add_angles = add_angles
        self.add_radial = add_radial
        self.iters = iters
        self.eps = eps

    @torch.no_grad()
    def _undistort_and_rays(self, H: int, W: int, K: torch.Tensor, dist: torch.Tensor,
                             device: torch.device, dtype: torch.dtype):
        # Force FP32 computation for numerical stability in extreme fisheye scenarios
        from torch.cuda.amp import autocast
        with autocast(enabled=False):
            # Convert inputs to FP32
            K = K.float()
            dist = dist.float()
            
            BN = K.shape[0]
            # Pixel grid
            u = torch.linspace(0, W - 1, W, device=device, dtype=torch.float32)
            v = torch.linspace(0, H - 1, H, device=device, dtype=torch.float32)
            uu, vv = torch.meshgrid(u, v)  # (W, H)
            uu = uu.t().unsqueeze(0).expand(BN, -1, -1)  # (BN, H, W)
            vv = vv.t().unsqueeze(0).expand(BN, -1, -1)

            fx = K[:, 0, 0].view(BN, 1, 1)
            fy = K[:, 1, 1].view(BN, 1, 1)
            cx = K[:, 0, 2].view(BN, 1, 1)
            cy = K[:, 1, 2].view(BN, 1, 1)

            k1 = dist[:, 0].view(BN, 1, 1)
            k2 = dist[:, 1].view(BN, 1, 1)
            p1 = dist[:, 2].view(BN, 1, 1)
            p2 = dist[:, 3].view(BN, 1, 1)
            xi = dist[:, 4].view(BN, 1, 1)

            # OpenCV omnidir pre-normalization with skew
            xd = (uu - cx) / (fx + self.eps)
            yd = (vv - cy) / (fy + self.eps)

            # Iterative undistortion to get (xu, yu) - corrected formula matching C++
            xu = xd.clone()
            yu = yd.clone()
            for _ in range(self.iters):
                r2 = xu * xu + yu * yu
                r4 = r2 * r2
                dist_factor = 1.0 + k1 * r2 + k2 * r4
                # Corrected tangential distortion terms to match C++ implementation
                dxu = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
                dyu = 2.0 * p2 * xu * yu + p1 * (r2 + 2.0 * yu * yu)
                xu = (xd - dxu) / (dist_factor + self.eps)
                yu = (yd - dyu) / (dist_factor + self.eps)

            # MEI sphere projection - corrected formula matching C++ implementation
            r2 = xu * xu + yu * yu
            # Solve quadratic equation: a*Zs^2 + b*Zs + c = 0
            a = r2 + 1.0
            b = 2.0 * xi * r2
            cc = r2 * xi * xi - 1.0
            
            # Discriminant with numerical stability
            discriminant = b * b - 4.0 * a * cc
            discriminant = torch.clamp(discriminant, min=self.eps)  # Ensure non-negative
            
            # Solve for Zs using quadratic formula (take positive root)
            zs = (-b + torch.sqrt(discriminant)) / (2.0 * a + self.eps)

            # Compute 3D coordinates
            factor = zs + xi
            xs = xu * factor
            ys = yu * factor
            # zs is already computed above
            
            # r_u2 = xu**2 + yu**2
            # sqrt_term_inner = torch.clamp(1 + (1 - xi**2) * r_u2, min=self.eps)
            # lambda_ = (xi + torch.sqrt(sqrt_term_inner)) / (1 + r_u2 + self.eps)
            
            # zs = lambda_ - xi
            # xs = lambda_ * xu
            # ys = lambda_ * yu


            ray = torch.stack([xs, ys, zs], dim=1)  # (BN, 3, H, W)
            ray = ray / (torch.norm(ray, dim=1, keepdim=True) + self.eps)
            
            ray = ray.to(dtype)
            return ray

    def forward(self, x: torch.Tensor, K: torch.Tensor, dist: torch.Tensor):
        BN, _, H, W = x.shape
        device, dtype = x.device, x.dtype

        ray = self._undistort_and_rays(H, W, K, dist, device, dtype)
        dx, dy, dz = ray[:, 0], ray[:, 1], ray[:, 2]

        # Channel 1: Theta (primary discriminator - angular coverage)
        rho = torch.sqrt(dx * dx + dy * dy)
        theta_unsigned = torch.atan2(rho, torch.abs(dz).clamp(min=self.eps))
        theta = torch.where(dz >= 0, theta_unsigned, -theta_unsigned)
        if hasattr(self, 'smooth_distortion_features') and self.smooth_distortion_features:
            kernel_size = 3
            padding = kernel_size // 2
            theta = F.avg_pool2d(theta.unsqueeze(1), kernel_size, stride=1, padding=padding).squeeze(1)
            dz = F.avg_pool2d(dz.unsqueeze(1), kernel_size, stride=1, padding=padding)
            rho = F.avg_pool2d(rho.unsqueeze(1), kernel_size, stride=1, padding=padding)
        
        dist_feats = {}
        if self.add_rays:
            dist_feats['rays'] = dz.unsqueeze(1)
        if self.add_angles:
            dist_feats['angles'] = theta.unsqueeze(1)
        if self.add_radial:
            dist_feats['radial'] = rho.unsqueeze(1)

        return dist_feats
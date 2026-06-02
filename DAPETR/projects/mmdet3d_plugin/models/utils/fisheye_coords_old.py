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

    def __init__(self, add_rays: bool = True, add_angles: bool = False, add_area: bool = False,
                                 iters: int = 20, eps: float = 1e-8):
        super().__init__()
        self.add_rays = add_rays
        self.add_angles = add_angles
        self.add_area = add_area
        self.iters = iters
        self.eps = eps
        # Debug controls via environment variables
        self.debug_dir = os.environ.get('FISH_FEATS_DEBUG', None)
        self.debug_max = int(os.environ.get('FISH_FEATS_DEBUG_MAX', '2'))
        self._debug_count = 0

    @torch.no_grad()
    def _undistort_and_rays(self, H: int, W: int, K: torch.Tensor, dist: torch.Tensor,
                             device: torch.device, dtype: torch.dtype):
        BN = K.shape[0]
        # Pixel grid
        u = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        v = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
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

        # Iterative undistortion to get (xu, yu)
        xu = xd.clone()
        yu = yd.clone()
        for _ in range(self.iters):
            r2 = xu * xu + yu * yu
            r4 = r2 * r2
            dist_factor = 1.0 + k1 * r2 + k2 * r4
            dxu = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
            dyu = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
            xu = (xd - dxu) / (dist_factor + self.eps)
            yu = (yd - dyu) / (dist_factor + self.eps)

        # MEI lift to sphere
        r_u2 = xu * xu + yu * yu
        sqrt_term = torch.sqrt(torch.clamp(1.0 + (1.0 - xi * xi) * r_u2, min=self.eps))
        lam = (xi + sqrt_term) / (1.0 + r_u2 + self.eps)
        xs = lam * xu
        ys = lam * yu
        zs = lam - xi

        ray = torch.stack([xs, ys, zs], dim=1)  # (BN, 3, H, W)
        ray = ray / (torch.norm(ray, dim=1, keepdim=True) + self.eps)
        return xu, yu, ray

    # @torch.no_grad()
    # def _area_distortion_from_rays(self, ray: torch.Tensor) -> torch.Tensor:
    #     """
    #     Robust local area distortion A(u,v) ≈ || ∂r/∂u × ∂r/∂v || for a unit-ray field r(u,v).
    #     Produces a log-standardized map with guards against NaNs/Inf and edge artifacts.
    #     """
    #     _, C, H, W = ray.shape
    #     assert C == 3, "ray must have 3 channels"

    #     # Sanitize and renormalize rays
    #     ray = torch.nan_to_num(ray, nan=0.0, posinf=0.0, neginf=0.0)
    #     ray = ray / (ray.norm(dim=1, keepdim=True).clamp(min=1e-6))

    #     # Light smoothing to reduce extreme local gradients
    #     ray_s = F.avg_pool2d(ray, kernel_size=3, stride=1, padding=1)

    #     # Sobel derivative filters
    #     kx = ray_s.new_tensor([[1.0, 0.0, -1.0],
    #                            [2.0, 0.0, -2.0],
    #                            [1.0, 0.0, -1.0]]).view(1, 1, 3, 3) / 8.0
    #     ky = ray_s.new_tensor([[1.0, 2.0, 1.0],
    #                            [0.0, 0.0, 0.0],
    #                            [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3) / 8.0

    #     # Reflect padding to avoid border artifacts; depthwise per-channel conv
    #     ray_pad = F.pad(ray_s, (1, 1, 1, 1), mode='reflect')
    #     du = F.conv2d(ray_pad, kx.expand(3, 1, 3, 3), padding=0, groups=3)
    #     dv = F.conv2d(ray_pad, ky.expand(3, 1, 3, 3), padding=0, groups=3)

    #     du_x, du_y, du_z = du[:, 0:1], du[:, 1:2], du[:, 2:3]
    #     dv_x, dv_y, dv_z = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]

    #     # Cross product magnitude ||du x dv|| with strict positivity
    #     cx = du_y * dv_z - du_z * dv_y
    #     cy = du_z * dv_x - du_x * dv_z
    #     cz = du_x * dv_y - du_y * dv_x
    #     sum_sq = cx.mul(cx) + cy.mul(cy) + cz.mul(cz)
    #     area = torch.sqrt(sum_sq.clamp_min(1e-12))

    #     # More stable area processing to reduce extreme gradients
    #     # Apply logarithm with offset to compress dynamic range
    #     area = torch.log1p(area * 10.0) / 10.0  # Scale down to reduce magnitude
        
    #     # Apply stronger smoothing before standardization
    #     area = F.avg_pool2d(area, kernel_size=5, stride=1, padding=2)

    #     # Per-image standardization with more conservative clamping
    #     mean = area.mean(dim=(2, 3), keepdim=True)
    #     std = area.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)  # Higher min std
    #     area = (area - mean) / std
    #     # area = area.clamp_(-3.0, 3.0)  # More conservative range

    #     # Final sanitize and ensure channel dimension
    #     area = torch.nan_to_num(area, nan=0.0, posinf=5.0, neginf=-5.0)
    #     if area.dim() == 3:
    #         area = area.unsqueeze(1)
    #     return area
    @torch.no_grad()
    def _area_distortion_from_rays(self, ray: torch.Tensor) -> torch.Tensor:
        """
        Compute stable solid angle per pixel for fisheye projection.
        Uses analytical approximation to avoid extreme numerical ranges.
        """
        _, C, H, W = ray.shape
        assert C == 3, "ray must have 3 channels"
        
        # Ensure rays are unit length and finite
        ray = torch.nan_to_num(ray, nan=0.0, posinf=0.0, neginf=0.0)
        ray_norm = ray.norm(dim=1, keepdim=True).clamp(min=1e-6)
        ray = ray / ray_norm
        
        dx, dy, dz = ray[:, 0], ray[:, 1], ray[:, 2]
        
        # Use analytical solid angle approximation: 1/cos³(θ) where cos(θ) = |dz|
        # This avoids the extreme numerical ranges from cross products
        dz_abs = torch.abs(dz).clamp(min=0.05)  # Avoid extreme values near horizon
        
        # Solid angle ∝ 1/cos³(θ) = 1/|dz|³
        solid_angle_analytical = 1.0 / (dz_abs * dz_abs * dz_abs)
        
        # print(f"Analytical solid angle min: {solid_angle_analytical.min().item():.6f}, max: {solid_angle_analytical.max().item():.6f}")
        
        # Apply log transformation with better numerical conditioning
        # Use log1p for better numerical stability near 1
        log_solid_angle = torch.log(solid_angle_analytical + 1e-6)
        
        # print(f"Analytical log solid angle min: {log_solid_angle.min().item():.6f}, max: {log_solid_angle.max().item():.6f}")
        
        # Apply strong smoothing to reduce sharp transitions
        log_solid_angle = log_solid_angle.unsqueeze(1)  # (BN, 1, H, W)
        log_solid_angle = F.avg_pool2d(log_solid_angle, kernel_size=7, stride=1, padding=3)
        
        # Normalize per image to zero mean
        mean_log = log_solid_angle.mean(dim=(2, 3), keepdim=True)
        centered_log = log_solid_angle - mean_log
        
        # Apply more aggressive range compression for stability
        # Use tanh to compress to [-1, 1] range, then scale
        compressed = torch.tanh(centered_log * 0.2)  # Compress input range
        area_distortion = compressed * 2.0  # Scale to [-2, 2]
        
        # print(f"Final area distortion min: {area_distortion.min().item():.6f}, max: {area_distortion.max().item():.6f}")
        
        return area_distortion

    def forward(self, x: torch.Tensor, K: torch.Tensor, dist: torch.Tensor):
        BN, _, H, W = x.shape
        device, dtype = x.device, x.dtype
        
        xu, yu, ray = self._undistort_and_rays(H, W, K, dist, device, dtype)
        dx, dy, dz = ray[:, 0], ray[:, 1], ray[:, 2]

        feats = []
        if self.add_rays:
            feats.append(ray)  # 3
        if self.add_angles:
            # Stabilized angle computation with gradient smoothing
            # Option 2: Signed theta (negative for backward rays)
            rho = torch.sqrt(dx * dx + dy * dy)
            theta_unsigned = torch.atan2(rho, torch.abs(dz).clamp(min=self.eps))
            theta = torch.where(dz >= 0, theta_unsigned, -theta_unsigned)
            phi = torch.atan2(dy, dx)
            
            # Apply light smoothing to reduce sharp gradients
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            
            # Smooth the trigonometric features to reduce training instability
            kernel_size = 3
            padding = kernel_size // 2
            cos_phi = F.avg_pool2d(cos_phi.unsqueeze(1), kernel_size, stride=1, padding=padding).squeeze(1)
            sin_phi = F.avg_pool2d(sin_phi.unsqueeze(1), kernel_size, stride=1, padding=padding).squeeze(1)
            cos_theta = F.avg_pool2d(cos_theta.unsqueeze(1), kernel_size, stride=1, padding=padding).squeeze(1)
            sin_theta = F.avg_pool2d(sin_theta.unsqueeze(1), kernel_size, stride=1, padding=padding).squeeze(1)
            
            ang = torch.stack([cos_phi, sin_phi, cos_theta, sin_theta], dim=1)
            feats.append(ang)  # 4
        if self.add_area:
            area = self._area_distortion_from_rays(ray)  # 1
        else:
            # Stabilized centrality feature with gradient smoothing
            area = (1.0 - dz).unsqueeze(1)
        # Apply smoothing and clamp to reasonable range
        area = F.avg_pool2d(area, 3, stride=1, padding=1)
        # area = torch.clamp(area, 0.0, 2.0)  # Prevent extreme values
        feats.append(area)  # 1
        if feats:
            out = torch.cat(feats, dim=1)
            # Optional debug: save a few maps and print stats
            if self.debug_dir and self._debug_count < self.debug_max:
                os.makedirs(self.debug_dir, exist_ok=True)
                with torch.no_grad():
                    b0 = 0  # first item in batch
                    # Save ray channels
                    def _to01(t):
                        t = t.detach().float().cpu().numpy()
                        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                        t_min, t_max = np.min(t), np.max(t)
                        if t_max - t_min < 1e-6:
                            return np.zeros_like(t)
                        return (t - t_min) / (t_max - t_min)
                    def _save(name, arr):
                        path = os.path.join(self.debug_dir, f"{self._debug_count:03d}_{name}.png")
                        if imageio is not None:
                            a = np.squeeze(arr)
                            # Convert to 2D grayscale or HxWx3 if possible
                            if a.ndim == 3:
                                # If channel-first 3xHxW, move to HxWx3
                                if a.shape[0] in (1, 3):
                                    a = np.transpose(a, (1, 2, 0))
                                # If channel-last HxWx1 -> squeeze
                                if a.shape[-1] == 1:
                                    a = np.squeeze(a, axis=-1)
                            # If still not 2D or valid 3D, reduce to 2D by taking first channel
                            if not (a.ndim == 2 or (a.ndim == 3 and a.shape[-1] in (3, 4))):
                                if a.ndim >= 3:
                                    a = a[..., 0]
                                a = np.squeeze(a)
                            img = (np.clip(a, 0, 1) * 255).astype(np.uint8)
                            imageio.imwrite(path, img)
                        else:
                            np.save(path.replace('.png', '.npy'), arr)
                    # dx, dy, dz
                    _save('dx', _to01(dx[b0:b0+1, ...]))
                    _save('dy', _to01(dy[b0:b0+1, ...]))
                    _save('dz', _to01(dz[b0:b0+1, ...]))
                    # 1-dz (centrality)
                    central = 1.0 - dz[b0:b0+1, ...]
                    _save('one_minus_dz', _to01(central))
                    # angle maps if present
                    if self.add_angles and not self.add_area:
                        theta = torch.atan2(torch.sqrt(dx * dx + dy * dy), torch.clamp(dz, min=self.eps))
                        phi = torch.atan2(dy, dx)
                        _save('theta', _to01(theta[b0:b0+1, ...]))
                        _save('phi', _to01(phi[b0:b0+1, ...]))
                    # area if present
                    area = feats[-1]  # last appended
                    _save('area', _to01(area[b0:b0+1, 0]))
                    # Print finite stats
                    def _stats(name, t):
                        finite = torch.isfinite(t)
                        count = int((~finite).sum().item())
                        t_f = t[finite]
                        tmin = float(t_f.min().item()) if t_f.numel() else float('nan')
                        tmax = float(t_f.max().item()) if t_f.numel() else float('nan')
                        print(f"[FisheyeCoordAug] {name}: non-finite={count}, min={tmin:.4f}, max={tmax:.4f}")
                    _stats('dx', dx)
                    _stats('dy', dy)
                    _stats('dz', dz)
                    if self.add_area:
                        _stats('area', feats[-1])
                self._debug_count += 1
            return out
        # No features requested; return empty tensor with correct spatial dims
        return x.new_zeros(BN, 0, H, W)

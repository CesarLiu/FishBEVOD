# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import math
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerPolarEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 with_distortion=False, with_polar=False,
                 radius_range=[1., 65., 1.], grid_res=0.8, **kwargs):

        super(BEVFormerPolarEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.with_distortion = with_distortion
        self.with_polar = with_polar
        self.radius_range = radius_range
        self.grid_res = grid_res
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Prepare transformation matrices based on camera model
        # Fisheye camera model
        lidar2cam = []
        K_matrices = []
        for img_meta in img_metas:
            lidar2cam.append(img_meta['lidar2cam'])
            K_matrices.append(img_meta['intrinsic'])
        lidar2cam = np.asarray(lidar2cam)
        K_matrices = np.asarray(K_matrices)
        # Convert to PyTorch tensors
        lidar2cam = reference_points.new_tensor(lidar2cam)  # (B, N, 4, 4)
        K_matrices = reference_points.new_tensor(K_matrices)  # Shape (B, N, 3, 3)
        num_cam = lidar2cam.size(1)
        if self.with_distortion:
            cam_dist_coeffs = []
            for img_meta in img_metas:
                cam_dist_coeffs.append(img_meta['dist_coeffs'])
            cam_dist_coeffs = np.asarray(cam_dist_coeffs)
            cam_dist_coeffs = reference_points.new_tensor(cam_dist_coeffs)  # Shape (B, N, 4)

        # Define point cloud range, e.g., pc_range = [xmin, ymin, zmin, xmax, ymax, zmax]
        reference_points = reference_points.clone()
        max_rho = (pc_range[3]**2 + pc_range[4]**2)**0.5
        rho = reference_points[..., 0:1] * max_rho
        phi = reference_points[..., 1:2] * 2 * math.pi - math.pi
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
        # Convert polar coordinates to Cartesian coordinates
        reference_points[..., 0:1] = rho * torch.cos(phi)
        reference_points[..., 1:2] = rho * torch.sin(phi)
        # Add homogeneous coordinate for transformations
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)
        # Permute to match expected shape and set dimensions
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]

        # Repeat reference points for each camera and reshape for matrix multiplication
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        bev_mask = torch.ones_like(reference_points[..., 2:3], dtype=torch.bool)
        
        lidar2cam = lidar2cam.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        K_expanded = K_matrices.view(1, B, num_cam, 1, 3, 3).repeat(D, 1, 1, num_query, 1, 1)
        # Transform reference points to camera coordinates using the lidar to camera transformation matrix
        reference_points_cam_3d = torch.matmul(lidar2cam.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5
        if self.with_distortion:
            # apply distortion coefficients
            # Extract X, Y, Z components
            last_dim = cam_dist_coeffs.shape[-1]
            cam_dist_coeffs = cam_dist_coeffs.view(1, B, num_cam, 1, last_dim).repeat(D, 1, 1, num_query, 1)
            # Separate distortion parameters for each camera 
            k1 = cam_dist_coeffs[..., 0] 
            k2 = cam_dist_coeffs[..., 1]    
            p1 = cam_dist_coeffs[..., 2]
            p2 = cam_dist_coeffs[..., 3]
            xi = cam_dist_coeffs[..., 4]
            # Extract X,Y,Z in camera frame
            X = reference_points_cam_3d[..., 0]
            Y = reference_points_cam_3d[..., 1]
            Z = reference_points_cam_3d[..., 2]

            # Unified omnidirectional normalization
            rho = torch.sqrt(torch.clamp(X * X + Y * Y + Z * Z, min=eps))
            denom = Z + xi * rho
            # Valid only when denom > 0
            valid_front = denom > eps
            x = X / torch.maximum(denom, torch.full_like(denom, eps))
            y = Y / torch.maximum(denom, torch.full_like(denom, eps))

            # Radial-tangential distortion
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * (r2 ** 2)
            x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

            # Project to pixels using intrinsics
            ones = torch.ones_like(x_d)
            distorted_points = torch.stack([x_d, y_d, ones], dim=-1).unsqueeze(-1)  # (...,3,1)
            reference_points_cam = torch.matmul(
                K_expanded.to(torch.float32), distorted_points.to(torch.float32)
            ).squeeze(-1)  # (...,3)

            # Visibility mask: denom positive and Z positive (optional)
            bev_mask = denom.unsqueeze(-1) > eps # & (Z.unsqueeze(-1) > eps)
            # Keep only u,v
            reference_points_cam = reference_points_cam[..., 0:2]
        else:
            # Transform reference points to camera coordinates using the lidar to camera transformation matrix
            reference_points_3d = reference_points_cam_3d[...,:3].to(torch.float32)
            reference_points_3d = reference_points_3d.unsqueeze(-1)
            reference_points_cam = torch.matmul(K_expanded.to(torch.float32),
                                                reference_points_3d).squeeze(-1)
            bev_mask = (reference_points_cam[..., 2:3] > eps)
            # Perspective division
            reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
                reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)


        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask
    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import cv2
from PIL import Image

def visualize_reference_points_on_images(reference_points_cam, bev_mask, img_metas, 
                                        sample_idx=0, max_points=1000, 
                                        save_path=None, point_size=1):
    """
    Visualize reference_points_cam projected on actual camera images.
    
    Args:
        reference_points_cam: Projected camera points [num_cam, bs, num_query, num_levels, 2]
        bev_mask: Visibility mask [num_cam, bs, num_query, num_levels]
        img_metas: Image metadata containing image paths and shapes
        sample_idx: Which sample in the batch to visualize
        max_points: Maximum number of points to visualize per camera
        save_path: Path to save the visualization
        point_size: Size of the plotted points
    """
    
    # Extract data for the specified sample
    ref_pts = reference_points_cam[:, sample_idx].cpu().numpy()  # [num_cam, num_query, num_levels, 2]
    mask = bev_mask[:, sample_idx].cpu().numpy()  # [num_cam, num_query, num_levels]
    
    num_cam, num_query, num_levels, _ = ref_pts.shape
    
    # Create subplot grid
    fig, axes = plt.subplots(2, (num_cam + 1) // 2, figsize=(20, 12))
    if num_cam <= 2:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    # Camera names for titles
    cam_names = ['CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT']
    if len(cam_names) < num_cam:
        cam_names.extend([f'CAM_{i}' for i in range(len(cam_names), num_cam)])
    
    # Color map for different levels
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta']
    
    for cam_idx in range(num_cam):
        # Load camera image
        img_path = img_metas[sample_idx]['filename'][cam_idx]
        img = Image.open(img_path)
        img_array = np.array(img)

        # Get image dimensions
        img_height, img_width = img_metas[sample_idx]['img_shape'][cam_idx][:2]

        # Plot image
        axes[cam_idx].imshow(img_array)
        axes[cam_idx].set_title(f'{cam_names[cam_idx]}', fontsize=14, fontweight='bold')
        
        # Process reference points for this camera
        cam_ref_pts = ref_pts[cam_idx]  # [num_query, num_levels, 2]
        cam_mask = mask[cam_idx]  # [num_query, num_levels]
        
        points_plotted = 0
        
        for level in range(num_levels):
            # Get valid points for this level
            valid_mask = cam_mask[:, level]
            valid_pts = cam_ref_pts[valid_mask, level, :]  # [num_valid, 2]
            
            if len(valid_pts) == 0:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            pixel_x = valid_pts[:, 0] * img_width
            pixel_y = valid_pts[:, 1] * img_height
            
            # Filter points within image bounds
            in_bounds = ((pixel_x >= 0) & (pixel_x < img_width) & 
                        (pixel_y >= 0) & (pixel_y < img_height))
            pixel_x = pixel_x[in_bounds]
            pixel_y = pixel_y[in_bounds]
            
            # Limit number of points for visibility
            if points_plotted + len(pixel_x) > max_points:
                remaining = max_points - points_plotted
                pixel_x = pixel_x[:remaining]
                pixel_y = pixel_y[:remaining]
            
            # Plot points
            if len(pixel_x) > 0:
                axes[cam_idx].scatter(pixel_x, pixel_y, 
                                    c=colors[level % len(colors)], 
                                    s=point_size, alpha=0.6, 
                                    label=f'Level {level}')
                points_plotted += len(pixel_x)
            
            if points_plotted >= max_points:
                break
        
        # Set axis properties
        axes[cam_idx].set_xlim(0, img_width)
        axes[cam_idx].set_ylim(img_height, 0)  # Flip y-axis for image coordinates
        axes[cam_idx].legend(loc='upper right', fontsize=8)
        axes[cam_idx].grid(True, alpha=0.3)
        
        # Add info text
        info_text = f'Points: {points_plotted}/{num_query * num_levels}'
        axes[cam_idx].text(0.02, 0.98, info_text, transform=axes[cam_idx].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          verticalalignment='top', fontsize=10)
    
    # Hide unused subplots
    for i in range(num_cam, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    
    plt.show()

def visualize_bev_grid_projection(reference_points_cam, bev_mask, bev_h, bev_w, 
                                sample_idx=0, save_path=None):
    """
    Visualize how BEV grid points project to different cameras.
    
    Args:
        reference_points_cam: Projected camera points [num_cam, bs, num_query, num_levels, 2]
        bev_mask: Visibility mask [num_cam, bs, num_query, num_levels]
        bev_h, bev_w: BEV grid dimensions
        sample_idx: Which sample to visualize
        save_path: Path to save the visualization
    """
    
    # Extract data
    ref_pts = reference_points_cam[:, sample_idx].cpu().numpy()  # [num_cam, num_query, num_levels, 2]
    mask = bev_mask[:, sample_idx].cpu().numpy()  # [num_cam, num_query, num_levels]
    
    num_cam = ref_pts.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, num_cam + 1, figsize=(25, 5))
    
    # Camera names
    cam_names = ['CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT']
    
    # Plot BEV grid visibility for each camera
    for cam_idx in range(num_cam):
        # Reshape mask to BEV grid format
        # Assuming queries are ordered as BEV grid points
        bev_visibility = mask[cam_idx, :, 0].reshape(bev_h, bev_w)  # Use first level
        
        # Plot BEV visibility
        im = axes[cam_idx].imshow(bev_visibility, cmap='RdYlBu', origin='lower')
        axes[cam_idx].set_title(f'{cam_names[cam_idx]}\nBEV Visibility', fontweight='bold')
        axes[cam_idx].set_xlabel('BEV Width')
        axes[cam_idx].set_ylabel('BEV Height')
        plt.colorbar(im, ax=axes[cam_idx], fraction=0.046, pad=0.04)
    
    # Plot combined visibility
    combined_visibility = np.any(mask[:, :, 0], axis=0).reshape(bev_h, bev_w)
    im = axes[-1].imshow(combined_visibility, cmap='RdYlBu', origin='lower')
    axes[-1].set_title('Combined\nBEV Visibility', fontweight='bold')
    axes[-1].set_xlabel('BEV Width')
    axes[-1].set_ylabel('BEV Height')
    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved BEV visualization: {save_path}")
    
    plt.show()


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 with_distortion=False, **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.with_distortion = with_distortion
        self.visualize_ref_points = False

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
    # Integration function to add to BEVFormerEncoder
    def visualize_reference_points_in_encoder(self, reference_points_cam, bev_mask, 
                                            img_metas, bev_h, bev_w, sample_idx=0):
        """
        Add this method to BEVFormerEncoder class for easy visualization.
        """
        if self.training:  # Only visualize during inference
            return
        
        # Initialize batch counter if not exists
        if not hasattr(self, '_vis_batch_counter'):
            self._vis_batch_counter = 0
        
        # Generate unique filenames for this batch
        batch_id = self._vis_batch_counter
        
        # Visualize points on camera images
        visualize_reference_points_on_images(
            reference_points_cam, bev_mask, img_metas,
            sample_idx=sample_idx,
            max_points=10000,
            save_path=f'reference_points_cameras_batch_{batch_id:04d}.png',
            point_size=2
        )
        
        # Visualize BEV grid projection
        visualize_bev_grid_projection(
            reference_points_cam, bev_mask, bev_h, bev_w,
            sample_idx=sample_idx,
            save_path=f'bev_grid_visibility_batch_{batch_id:04d}.png'
        )
        
        # Increment batch counter for next call
        self._vis_batch_counter += 1
        
        print(f"Saved batch {batch_id} visualizations")
    # # This function must use fp32!!!
    # @force_fp32(apply_to=('reference_points', 'img_metas'))
    # def point_sampling(self, reference_points, pc_range,  img_metas):
    #     # NOTE: close tf32 here.
    #     allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    #     torch.backends.cuda.matmul.allow_tf32 = False
    #     torch.backends.cudnn.allow_tf32 = False

    #     lidar2img = []
    #     for img_meta in img_metas:
    #         lidar2img.append(img_meta['lidar2img'])
    #     lidar2img = np.asarray(lidar2img)
    #     lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    #     reference_points = reference_points.clone()

    #     reference_points[..., 0:1] = reference_points[..., 0:1] * \
    #         (pc_range[3] - pc_range[0]) + pc_range[0]
    #     reference_points[..., 1:2] = reference_points[..., 1:2] * \
    #         (pc_range[4] - pc_range[1]) + pc_range[1]
    #     reference_points[..., 2:3] = reference_points[..., 2:3] * \
    #         (pc_range[5] - pc_range[2]) + pc_range[2]

    #     reference_points = torch.cat(
    #         (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    #     reference_points = reference_points.permute(1, 0, 2, 3)
    #     D, B, num_query = reference_points.size()[:3]
    #     num_cam = lidar2img.size(1)

    #     reference_points = reference_points.view(
    #         D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

    #     lidar2img = lidar2img.view(
    #         1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    #     reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
    #                                         reference_points.to(torch.float32)).squeeze(-1)
    #     eps = 1e-5

    #     bev_mask = (reference_points_cam[..., 2:3] > eps)
    #     reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
    #         reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    #     reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    #     reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    #     bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
    #                 & (reference_points_cam[..., 1:2] < 1.0)
    #                 & (reference_points_cam[..., 0:1] < 1.0)
    #                 & (reference_points_cam[..., 0:1] > 0.0))
    #     if digit_version(TORCH_VERSION) >= digit_version('1.8'):
    #         bev_mask = torch.nan_to_num(bev_mask)
    #     else:
    #         bev_mask = bev_mask.new_tensor(
    #             np.nan_to_num(bev_mask.cpu().numpy()))

    #     reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    #     bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    #     torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    #     torch.backends.cudnn.allow_tf32 = allow_tf32

    #     return reference_points_cam, bev_mask
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
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
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
            # Apply MEI (Unified Omnidirectional) projection with distortion
            # cam_dist_coeffs expected per-cam; support either [k1,k2,p1,p2,xi] or [k1,k2,p1,p2,k3,xi,...]
            last_dim = cam_dist_coeffs.shape[-1]
            # Broadcast to (D, B, N, Q, C)
            cam_dist_coeffs = cam_dist_coeffs.view(1, B, num_cam, 1, last_dim).repeat(D, 1, 1, num_query, 1)
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


        # Normalize by per-camera image size to [0,1]
        # Collect per-batch, per-cam (w,h)
        ws = []
        hs = []
        for b in range(B):
            w_row = []
            h_row = []
            for c in range(num_cam):
                h_row.append(img_metas[b]['img_shape'][c][0])
                w_row.append(img_metas[b]['img_shape'][c][1])
            hs.append(h_row)
            ws.append(w_row)
        ws = reference_points.new_tensor(ws)  # (B, N)
        hs = reference_points.new_tensor(hs)  # (B, N)
        # Expand to (D, B, N, Q, 1)
        ws = ws.view(1, B, num_cam, 1, 1).repeat(D, 1, 1, num_query, 1)
        hs = hs.view(1, B, num_cam, 1, 1).repeat(D, 1, 1, num_query, 1)
        reference_points_cam[..., 0:1] = reference_points_cam[..., 0:1] / torch.maximum(ws, torch.ones_like(ws))
        reference_points_cam[..., 1:2] = reference_points_cam[..., 1:2] / torch.maximum(hs, torch.ones_like(hs))

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
        # Add visualization call here
        if not self.training and hasattr(self, 'visualize_ref_points') and self.visualize_ref_points:
            self.visualize_reference_points_in_encoder(
                reference_points_cam, bev_mask, kwargs['img_metas'], 
                bev_h, bev_w, sample_idx=0
            )
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


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query




from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


@TRANSFORMER_LAYER.register_module()
class MM_BEVFormerLayer(MyCustomBaseTransformerLayer):
    """multi-modality fusion layer.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 lidar_cross_attn_layer=None,
                 **kwargs):
        super(MM_BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.cross_model_weights = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True) 
        if lidar_cross_attn_layer:
            self.lidar_cross_attn_layer = build_attention(lidar_cross_attn_layer)
            # self.cross_model_weights+=1
        else:
            self.lidar_cross_attn_layer = None


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                depth=None,
                depth_z=None,
                lidar_bev=None,
                radar_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    lidar_bev=lidar_bev,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                new_query1 = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    depth=depth,
                    lidar_bev=lidar_bev,
                    depth_z=depth_z,
                    **kwargs)

                if self.lidar_cross_attn_layer:
                    bs = query.size(0)
                    new_query2 = self.lidar_cross_attn_layer(
                        query,
                        lidar_bev,
                        lidar_bev,
                        reference_points=ref_2d[bs:],
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        )
                query = new_query1 * self.cross_model_weights + (1-self.cross_model_weights) * new_query2
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

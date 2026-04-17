# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast


@NECKS.register_module(force=True)
class PolarLSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
    ):
        super(PolarLSSViewTransformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)      # (D, fH, fW, 3)  3:(u, v, d)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z

    def create_grid_infos(self, azimuth, radius, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [azimuth, radius, z]])     # (min_r, min_d, min_z)
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [azimuth, radius, z]])        # (dr, dd, dz)
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [azimuth, radius, z]])     # (Dr, Dd, Dz)

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        Returns:
            frustum: (D, fH, fW, 3)  3:(u, v, d)
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)      # (D, fH, fW)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)

        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)      # (D, fH, fW)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)      # (D, fH, fW)

        return torch.stack((x, y, d), -1)    # (D, fH, fW, 3)  3:(u, v, d)

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                     bda, dist_coeffs = None):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            sensor2ego (torch.Tensor): Transformation from camera coordinate system to
                ego coordinate system in shape (B, N_cams, 4, 4).
            ego2global (torch.Tensor): Translation from ego coordinate system to
                global coordinate system in shape (B, N_cams, 4, 4).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).
            bda (torch.Tensor): Transformation in bev. (B, 3, 3)

        Returns:
            torch.tensor: Point coordinates in shape (B, N, D, fH, fW, 3)
        """
        B, N, _, _ = sensor2ego.shape
        eps = 1e-6

        # post-transformation
        # (D, fH, fW, 3) - (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        # (B, N_, D, fH, fW, 3, 1)  3: (du, dv, d)
        if dist_coeffs is None:
            points = torch.cat(
                (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
            # R_{c->e} @ K^-1
            combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
            # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
            # --> (B, N, D, fH, fW, 3)
            points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
            # (B, N, D, fH, fW, 3) + (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
            points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)
        else:
            # apply distortion
            # Step 1: Apply inverse intrinsics to get normalized coordinates
            inv_intrins = torch.inverse(cam2imgs)
            u = points[..., 0, 0]       # (B,N,D,H,W)
            v = points[..., 1, 0]       # (B,N,D,H,W)
            d_z = points[..., 2, 0]     # (B,N,D,H,W)  Z-depth in camera frame
            ones = torch.ones_like(u)
            pix = torch.stack([u, v, ones], dim=-1).unsqueeze(-1)  # (...,3,1)
            points_norm = inv_intrins.view(B, N, 1, 1, 1, 3, 3).matmul(pix).squeeze(-1)

            # Extract normalized coordinates u', v'
            u_prime = points_norm[:, :, :, :, :, 0]
            v_prime = points_norm[:, :, :, :, :, 1]
            # d = points_norm[:, :, :, :, :, 2]  # Depth
            
            # Step 3: Extract distortion coefficients
            k1, k2, p1, p2, xi = (
                dist_coeffs[..., 0].view(B, N, 1, 1, 1),
                dist_coeffs[..., 1].view(B, N, 1, 1, 1),
                dist_coeffs[..., 2].view(B, N, 1, 1, 1),
                dist_coeffs[..., 3].view(B, N, 1, 1, 1),
                dist_coeffs[..., 4].view(B, N, 1, 1, 1)
            )
            # Step 4: Calculate theta from rho using Newton's method
            # Initialize theta with rho (good first approximation for small angles)
            xu, yu = u_prime.clone(), v_prime.clone()
            for _ in range(5):
                r2 = xu**2 + yu**2
                dist_factor = 1 + k1 * r2 + k2 * r2**2
                dxu = 2 * p1 * xu * yu + p2 * (r2 + 2 * xu**2)
                dyu = p1 * (r2 + 2 * yu**2) + 2 * p2 * xu * yu
                xu = (u_prime - dxu) / (dist_factor + eps)
                yu = (v_prime - dyu) / (dist_factor + eps)

            # 10.12.2025 correction MEI sphere projection - corrected formula matching C++ implementation
            r2 = xu * xu + yu * yu
            # Solve quadratic equation: a*Zs^2 + b*Zs + c = 0
            a = r2 + 1.0
            b = 2.0 * xi * r2
            cc = r2 * xi * xi - 1.0
            
            # Discriminant with numerical stability
            discriminant = b * b - 4.0 * a * cc
            discriminant = torch.clamp(discriminant, min=eps)  # Ensure non-negative
            
            # Solve for Zs using quadratic formula (take positive root)
            zs = (-b + torch.sqrt(discriminant)) / (2.0 * a)
            
            # Compute 3D coordinates
            factor = zs + xi

            Xc, Yc, Zc = xu * factor * d_z, yu * factor * d_z, zs * d_z
            # Create rays with proper dimensions
            points_3d = torch.stack([Xc, Yc, Zc], dim=-1)
            # Step 6: Apply inverse camera-to-lidar transformation
            points = sensor2ego[..., :3, :3].view(B, N, 1, 1, 1, 3, 3).matmul(points_3d.unsqueeze(-1)).squeeze(-1)
            points += sensor2ego[..., :3, 3].view(B, N, 1, 1, 1, 3)

        # (B, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1) --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = bda[:, :3, :3].view(B, 1, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)).squeeze(-1)
        points += bda[:, :3, 3].view(B, 1, 1, 1, 1, 3)

        return points

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):
        """
        Args:
            coor: (B, N, D, fH, fW, 3)
            depth: (B, N, D, fH, fW)
            feat: (B, N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
        """
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[1]),
                int(self.grid_size[0])
            ]).to(feat)     # (B, C, Dz, Dd, Dr)
            dummy = torch.cat(dummy.unbind(dim=2), 1)   # (B, C*Dz, Dd, Dr)
            return dummy

        feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])       # (B, Dz, Dy, Dx, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)    # (B, C, Dz, Dy, Dx)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)     # (B, C*Dz, Dy, Dx)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).
        Returns:
            tuple[torch.tensor]:
                ranks_bev: Rank of the voxel that a point is belong to in shape (N_points, ),
                    rank介于(0, B*Dx*Dy*Dz-1).
                ranks_depth: Reserved index of points in the depth space in shape (N_Points),
                    rank介于(0, B*N*D*fH*fW-1).
                ranks_feat: Reserved index of points in the feature space in shape (N_Points),
                    rank介于(0, B*N*fH*fW-1).
                interval_starts: (N_pillar, )  该pillar的起始位置  (N_pillar, )
                interval_lengths: (N_pillar, )  该pillar包含points的数量 (N_pillar, )
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(
            0, num_points, dtype=torch.int, device=coor.device)    # (B*N*D*H*W, ), [0, 1, ..., B*N*D*fH*fW-1]
        ranks_feat = torch.arange(
            0, num_points // D, dtype=torch.int, device=coor.device)   # [0, 1, ...,B*N*fH*fW-1]
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()     # (B*N*D*fH*fW, )

        # convert coordinate into the voxel space
        # ((B, N, D, fH, fW, 3) - (3, )) / (3, ) --> (B, N, D, fH, fW, 3)   3:(x, y, z)  polar grid坐标.
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor)).long()
        coor = coor.view(num_points, 3)      # (B, N, D, fH, fW, 3) --> (B*N*D*fH*fW, 3)
        # (B, N*D*fH*fW) --> (B*N*D*fH*fW, 1)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)      # (B*N*D*fH*fW, 4)   4: (x, y, z, batch_id)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None

        # (N_points, 4), (N_points, ), (N_points, )
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        # (N_points, ), (N_points, ), (N_points, )
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]  # 边界点=1, 其余为0（pillar id发生变化）
        interval_starts = torch.where(kept)[0].int()    # 该pillar的起始位置  (N_pillar, )
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]     # pillar包含points的数量  (N_pillar, )
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_ego_coor(*input[1:8])       # (B, N, D, fH, fW, 3)
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        def cart2polar(coord):
            """
            Args:
                coord: (B, N, D, H, W, 3)   3: (x, y, z)
            Returns:
                polar_coord: (B, N, D, H, W, 3)   3: (r, d, z)
            """
            phi = torch.atan2(coord[..., 1], coord[..., 0])  # (-pi, pi)
            rho = torch.sqrt(coord[..., 0] ** 2 + coord[..., 1] ** 2)
            return torch.stack((phi, rho, coord[..., 2]), dim=-1)

        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)      # (B, N, C, fH, fW)
            feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C)
            depth = depth.view(B, N, self.D, H, W)      # (B, N, D, fH, fW)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])   # (B, Dz, Dy, Dx, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)   # (B, C, Dz, Dy, Dx)

            bev_feat = bev_feat.squeeze(2)      # (B, C, Dy, Dx)
        else:
            coor = self.get_ego_coor(*input[1:8])   # (B, N, D, fH, fW, 3)
            coor = cart2polar(coor)     # (B, N, D, fH, fW, 3)
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))      # (B, C*Dz(=1), Dy, Dx)

        return bev_feat, depth

    def view_transform(self, input, depth, tran_feat):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, C, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat)

    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        x = input[0]    # (B, N, C_in, fH, fW)
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)      # (B*N, C_in, fH, fW)

        # (B*N, C_in, fH, fW) --> (B*N, D+C, fH, fW)
        x = self.depth_net(x)
        depth_digit = x[:, :self.D, ...]    # (B*N, D, fH, fW)
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]    # (B*N, C, fH, fW)
        depth = depth_digit.softmax(dim=1)
        bev_feat, depth = self.view_transform(input, depth, tran_feat)

        return bev_feat, depth

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None
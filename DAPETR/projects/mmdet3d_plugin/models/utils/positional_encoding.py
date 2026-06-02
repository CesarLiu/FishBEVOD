# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule

@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding3D(BaseModule):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding3D, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask, coords=None):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if coords is not None:
            if coords.dim() == 4:
                BN, C, H, W = coords.shape
                B, N = 1, BN
                coords_5d = coords.view(B, N, C, H, W)
            else:
                B, N, C, H, W = coords.shape
                coords_5d = coords

            # Expect at least Xc, Yc
            assert C >= 2, f'coords must have at least 2 channels (Xc, Yc), got {C}'

            device = coords_5d.device

            Xc = coords_5d[:, :, 0, :, :]
            Yc = coords_5d[:, :, 1, :, :]

            # Polar conversion
            r = torch.sqrt(torch.clamp(Xc * Xc + Yc * Yc, min=self.eps))  # [B,N,H,W]
            theta = torch.atan2(Yc, Xc)  # in [-pi, pi]
            
        if self.normalize:
            n_embed = (n_embed + self.offset) / \
                      (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, :, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_n = n_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, N, H, W = mask.size()
        pos_n = torch.stack(
            (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        if coords is not None:
            r_embed = torch.clamp(r, 0.0, 1.0) * self.scale
            pos_r = r_embed[:, :, :, :, None] / dim_t
            pos_theta = theta[:, :, :, :, None] / dim_t
            pos_r = torch.stack(
                (pos_r[:, :, :, :, 0::2].sin(), pos_r[:, :, :, :, 1::2].cos()),
                dim=4).view(B, N, H, W, -1)
            pos_theta = torch.stack(
                (pos_theta[:, :, :, :, 0::2].sin(), pos_theta[:, :, :, :, 1::2].cos()),
                dim=4).view(B, N, H, W, -1)
            pos = torch.cat((pos_n, pos_y, pos_x, pos_r, pos_theta), dim=4).permute(0, 1, 4, 2, 3)
        else:
            pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class DistortionAwarePolarPositionalEncoding3D(BaseModule):
    """Polar positional encoding aware of camera distortion.

    This encoder assumes the MEI (or other) distortion correction has already
    been applied upstream. It expects camera-centric image-plane coordinates
    (Xc, Yc) and optionally a third channel encoding a per-view index/value
    (n). It converts (Xc, Yc) to polar coordinates (r, theta) and applies a
    sinusoidal encoding to (r, theta, n) analogous to SinePositionalEncoding3D.

    Inputs:
      - coords: Tensor with shape
          [B, N, C, H, W] or [B*N, C, H, W], where C is 2 or 3:
            coords[..., 0, :, :] = Xc
            coords[..., 1, :, :] = Yc
            coords[..., 2, :, :] = optional n channel (already normalized)

    Outputs:
      - pos: sinusoidal embedding with shape
          [B, N, num_feats*3, H, W] for 5D input, or
          [B*N, num_feats*3, H, W] for 4D input.

    Args:
        num_feats (int): feature dimension per axis before sine/cosine.
        temperature (int): temperature for scaling frequencies.
        normalize (bool): whether to normalize r to [0, 1] per spatial map.
        scale_r (float): scale for the radius channel (used when normalize=True).
        scale_theta (float): scale for the theta channel (angle in radians).
        eps (float): numerical stability epsilon.
        offset (float): offset added when normalizing r.
        init_cfg (dict or list[dict], optional): mmdet-style init config.
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale_r=2 * math.pi,
                 scale_theta=1.0,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(DistortionAwarePolarPositionalEncoding3D, self).__init__(init_cfg)
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale_r = scale_r
        self.scale_theta = scale_theta
        self.eps = eps
        self.offset = offset

    def _sinusoidal(self, value: torch.Tensor, dim_t: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal encoding along the last dim via broadcasting.

        Expects value shape [..., H, W] and returns [..., H, W, num_feats].
        """
        v = value[..., None] / dim_t  # [..., H, W, num_feats]
        s = torch.stack((v[..., 0::2].sin(), v[..., 1::2].cos()), dim=-1)
        # fold the last two dims back into num_feats
        return s.flatten(-2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # Unify to 5D [B, N, C, H, W]
        assert coords.dim() in (4, 5), \
            f'coords must be 4D or 5D, got {coords.dim()}D'
        if coords.dim() == 4:
            BN, C, H, W = coords.shape
            B, N = 1, BN
            coords_5d = coords.view(B, N, C, H, W)
        else:
            B, N, C, H, W = coords.shape
            coords_5d = coords

        # Expect at least Xc, Yc
        assert C >= 2, f'coords must have at least 2 channels (Xc, Yc), got {C}'

        device = coords_5d.device

        Xc = coords_5d[:, :, 0, :, :]
        Yc = coords_5d[:, :, 1, :, :]

        # Polar conversion
        r = torch.sqrt(torch.clamp(Xc * Xc + Yc * Yc, min=self.eps))  # [B,N,H,W]
        theta = torch.atan2(Yc, Xc)  # in [-pi, pi]

        # Optional normalization for r per (B,N) map
        if self.normalize:
            r = torch.clamp(r, 0.0, 1.0) * self.scale_r
        else:
            # scale directly (assumes r already in a comparable range)
            r = r * self.scale_r

        # For theta, keep radians and scale if requested (periodic by nature)
        theta = theta * self.scale_theta

        # synthesize a simple per-view ramp in [0,1]
        view_ids = torch.arange(N, device=device, dtype=torch.float32)
        if N > 1:
            view_ids = (view_ids - view_ids.min()) / (view_ids.max() - view_ids.min())
        else:
            view_ids = view_ids.fill_(0.)
        n_embed = (view_ids * self.scale_r).view(1, N, 1, 1).expand(B, N, H, W)

        # Build frequency vector
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        # Compute encodings -> [B,N,H,W,num_feats] each
        pos_r = self._sinusoidal(r, dim_t)
        pos_theta = self._sinusoidal(theta, dim_t)
        pos_n = self._sinusoidal(n_embed, dim_t)

        # Concatenate along feature dim and permute to channels-first
        pos = torch.cat((pos_r, pos_theta, pos_n), dim=-1)  # [B,N,H,W,3*num_feats]
        pos = pos.permute(0, 1, 4, 2, 3).contiguous()  # [B,N,3F,H,W]

        # Return in the same rank as input
        if coords.dim() == 4:
            return pos.view(N, 3 * self.num_feats, H, W)  # [B*N, 3F, H, W] with B=1
        return pos

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale_r={self.scale_r}, '
        repr_str += f'scale_theta={self.scale_theta}, '
        repr_str += f'eps={self.eps})'
        return repr_str
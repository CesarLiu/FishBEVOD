# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
import numpy as np
from mmcv.cnn import xavier_init, constant_init, kaiming_init
import math
from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.robust_film import MultiModalDistortionFiLM_FP16
def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    # since x,y,z convert to polar coordinates rho, theta, z
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
class FPEFiLMLayer(nn.Module):
    """Dedicated FiLM layer for Feature-guided Positional Encoding where 
    features and condition have the same dimensions"""
    
    def __init__(self, feature_dim, reduction_ratio=16, temperature_init=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Shared encoder for both gamma and beta generation
        intermediate_dim = max(feature_dim // reduction_ratio, 32)
        
        # Condition encoder - processes image features to generate conditioning signal
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, intermediate_dim, 1, bias=False),
            nn.BatchNorm2d(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_dim, intermediate_dim, 1, bias=False),
            nn.BatchNorm2d(intermediate_dim),
            nn.ReLU(inplace=True)
        )
        
        # Gamma network - multiplicative modulation
        self.gamma_net = nn.Sequential(
            nn.Conv2d(intermediate_dim, feature_dim, 1, bias=True)
        )
        
        # Beta network - additive modulation
        self.beta_net = nn.Sequential(
            nn.Conv2d(intermediate_dim, feature_dim, 1, bias=True)
        )
        
        # Learnable temperature for controlling modulation strength
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        
        # Residual weight for stability
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self.init_weights()
        
    def init_weights(self):
        """Conservative initialization to prevent gradient explosion"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Very small initialization for output layers
                if m in [self.gamma_net[-1], self.beta_net[-1]]:
                    m.weight.data *= 0.01
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Initialize gamma net bias to produce gamma ≈ 1.0 initially
        if self.gamma_net[-1].bias is not None:
            nn.init.constant_(self.gamma_net[-1].bias, 0)
            
    def forward(self, position_features, image_features, debug=False):
        """
        Args:
            position_features: Position embeddings [B*N, feature_dim, H, W]
            image_features: Image features (condition) [B*N, feature_dim, H, W]
        Returns:
            Modulated position features [B*N, feature_dim, H, W]
        """
        # Safety checks
        if torch.isnan(position_features).any() or torch.isinf(position_features).any():
            if debug:
                print("NaN/inf detected in position_features")
            return position_features
            
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            if debug:
                print("NaN/inf detected in image_features")
            return position_features
            
        try:
            # Encode image features to generate conditioning signal
            condition_encoded = self.condition_encoder(image_features)  # [B*N, intermediate_dim, H, W]
            
            # Generate modulation parameters
            gamma_raw = self.gamma_net(condition_encoded)  # [B*N, feature_dim, H, W]
            beta_raw = self.beta_net(condition_encoded)    # [B*N, feature_dim, H, W]
            
            # Apply temperature scaling and safe activation
            temp = torch.clamp(self.temperature, min=0.01, max=1.0)
            
            # Safe gamma: 1 + small_perturbation (centered around 1 for multiplicative identity)
            gamma = 1.0 + torch.tanh(gamma_raw * temp) * 0.4  # Range: [0.6, 1.4]

            # Safe beta: small additive perturbation (centered around 0 for additive identity)
            beta = torch.tanh(beta_raw * temp) * 0.2  # Range: [-0.2, 0.2]
            
            # Clamp to prevent extreme values
            gamma = torch.clamp(gamma, min=0.5, max=2.0)
            beta = torch.clamp(beta, min=-0.3, max=0.3)
            
            # Safety checks on modulation parameters
            if torch.isnan(gamma).any() or torch.isinf(gamma).any():
                if debug:
                    print("NaN/inf detected in gamma")
                gamma = torch.ones_like(gamma)
            if torch.isnan(beta).any() or torch.isinf(beta).any():
                if debug:
                    print("NaN/inf detected in beta")
                beta = torch.zeros_like(beta)
            
            # Apply FiLM: γ * features + β
            modulated_features = gamma * position_features + beta
            
            # Add learnable residual connection for stability
            residual_weight = torch.clamp(self.residual_weight, min=0.0, max=0.5)
            modulated_features = (1.0 - residual_weight) * modulated_features + residual_weight * position_features
            
            # Final safety check
            if torch.isnan(modulated_features).any() or torch.isinf(modulated_features).any():
                if debug:
                    print("NaN/inf detected in final output")
                return position_features
                
            return modulated_features
            
        except Exception as e:
            if debug:
                print(f"FPEFiLM forward failed: {e}")
            return position_features
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation Layer with improved numerical stability"""
    def __init__(self, condition_dim, feature_dim, reduction_ratio=16):
        super().__init__()
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim
        
        # Use smaller intermediate dimension for stability
        intermediate_dim = max(feature_dim // reduction_ratio, 32)
        
        # More conservative gamma network with better initialization
        self.gamma_net = nn.Sequential(
            nn.Conv2d(condition_dim, intermediate_dim, 1, bias=False),
            nn.BatchNorm2d(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_dim, feature_dim, 1, bias=True)
        )
        
        # More conservative beta network
        self.beta_net = nn.Sequential(
            nn.Conv2d(condition_dim, intermediate_dim, 1, bias=False),
            nn.BatchNorm2d(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_dim, feature_dim, 1, bias=True)
        )
        
        # Learnable temperature for controlling modulation strength
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        self.init_weights()
        
    def init_weights(self):
        """Conservative initialization to prevent gradient explosion"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Very small initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.weight.size(0) == self.feature_dim:  # Output layer
                    m.weight.data *= 0.01  # Very small scale
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, features, condition):
        """
        Args:
            features: Image features [B*N, C, H, W]
            condition: 3D position embeddings [B*N, condition_dim, H, W]
        Returns:
            Modulated features [B*N, C, H, W]
        """
        # Check for NaN/inf in inputs
        if torch.isnan(features).any() or torch.isinf(features).any():
            return features
        if torch.isnan(condition).any() or torch.isinf(condition).any():
            return features
            
        try:
            # Generate modulation parameters with temperature control
            gamma_raw = self.gamma_net(condition)  # [B*N, C, H, W]
            beta_raw = self.beta_net(condition)    # [B*N, C, H, W]
            
            # Apply temperature scaling and safe activation
            temp = torch.clamp(self.temperature, min=0.01, max=1.0)
            
            # Safe gamma: 1 + small_perturbation (multiplicative identity around 1)
            gamma = 1.0 + torch.tanh(gamma_raw * temp) * 0.4  # Range: [0.8, 1.2]
            
            # Safe beta: small additive perturbation (additive identity around 0)
            beta = torch.tanh(beta_raw * temp) * 0.1  # Range: [-0.1, 0.1]
            
            # Clamp to prevent extreme values
            gamma = torch.clamp(gamma, min=0.5, max=2.0)
            beta = torch.clamp(beta, min=-0.5, max=0.5)
            
            # Check for NaN/inf in modulation parameters
            if torch.isnan(gamma).any() or torch.isinf(gamma).any():
                print("NaN/inf detected in gamma")
                gamma = torch.ones_like(gamma)
            if torch.isnan(beta).any() or torch.isinf(beta).any():
                print("NaN/inf detected in beta")
                beta = torch.zeros_like(beta)
            
            # Apply FiLM with residual connection for stability
            modulated_features = gamma * features + beta
            
            # Add small residual connection to original features for stability
            modulated_features = 0.9 * modulated_features + 0.1 * features
            
            # Final safety check
            if torch.isnan(modulated_features).any() or torch.isinf(modulated_features).any():
                return features
                
            return modulated_features
            
        except Exception as e:
            # Fallback to identity if anything goes wrong
            print(f"FiLM forward failed: {e}")
            return features

@HEADS.register_module()
class PETRHeadPolar4Nusc(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 with_fpe=True,
                 use_fpe_film=False,
                 with_distortion=False,
                 with_film=False,
                 with_polar_pe=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is PETRHeadPolar4Nusc):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
            #     'The regression iou weight for loss and matcher should be' \
            #     'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.with_distortion = with_distortion
        self.with_film = with_film

        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        self.with_fpe = with_fpe
        self.use_fpe_film = use_fpe_film
        self.with_polar_pe = with_polar_pe
        super(PETRHeadPolar4Nusc, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        # self.activate = build_activation_layer(self.act_cfg)
        # if self.with_multiview or not self.with_position:
        #     self.positional_encoding = build_positional_encoding(
        #         positional_encoding)
        self.positional_encoding = build_positional_encoding(
                positional_encoding)
        self.transformer = build_transformer(transformer)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            if self.with_polar_pe:
                pe_dims = 5
            else:
                pe_dims = 3
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*pe_dims//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        if self.with_fpe:
            if self.use_fpe_film:
                self.fpe = FPEFiLMLayer(self.embed_dims)
            else:
                self.fpe = SELayer(self.embed_dims)
        if self.with_film:
            self.film = FiLMLayer(
                condition_dim=self.embed_dims,  # 3D position embedding dimension
                feature_dim=self.embed_dims     # Image feature dimension
            )
    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]

        if self.with_distortion:
            coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
            coords = coords.view(1, 1, W, H, D, 3).repeat(B, N, 1, 1, 1, 1) # B, N, W, H, D, 3
            intrinsics = []
            extrinsics = []
            dist_coeffs = []
            for img_meta in img_metas:
                intrinsics.append(img_meta['intrinsics'])
                extrinsics.append(img_meta['extrinsics'])
                dist_coeffs.append(img_meta['dist_coeffs'])
            intrinsics = coords.new_tensor(np.asarray(intrinsics)) # (B, N, 4, 4)
            extrinsics = coords.new_tensor(np.asarray(extrinsics)) # (B, N, 4, 4)
            dist_coeffs = coords.new_tensor(np.asarray(dist_coeffs)) # (B, N, 5)
            intrinsics = intrinsics.view(B, N, 1, 1, 1, 4, 4)
            cam2lidar = torch.inverse(extrinsics).view(B, N, 1, 1, 1, 4, 4)
            
            k1 = dist_coeffs[..., 0].view(B, N, 1, 1, 1)
            k2 = dist_coeffs[..., 1].view(B, N, 1, 1, 1)
            p1 = dist_coeffs[..., 2].view(B, N, 1, 1, 1)
            p2 = dist_coeffs[..., 3].view(B, N, 1, 1, 1)
            xi = dist_coeffs[..., 4].view(B, N, 1, 1, 1)

            fx = intrinsics[..., 0, 0]
            fy = intrinsics[..., 1, 1]
            cx = intrinsics[..., 0, 2]
            cy = intrinsics[..., 1, 2]
            u, v, d = coords[..., 0], coords[..., 1], coords[..., 2]
            xd = (u - cx) / fx
            yd = (v - cy) / fy
            xu, yu = xd.clone(), yd.clone()
            for _ in range(5):
                r2 = xu**2 + yu**2
                dist_factor = 1 + k1 * r2 + k2 * r2**2
                dxu = 2 * p1 * xu * yu + p2 * (r2 + 2 * xu**2)
                dyu = p1 * (r2 + 2 * yu**2) + 2 * p2 * xu * yu
                xu = (xd - dxu) / (dist_factor + eps)
                yu = (yd - dyu) / (dist_factor + eps)

            # MEI sphere projection - corrected formula matching C++ implementation
            r2 = xu * xu + yu * yu
            # Solve quadratic equation: a*Zs^2 + b*Zs + c = 0
            a = r2 + 1.0
            b = 2.0 * xi * r2
            cc = r2 * xi * xi - 1.0
            
            # Discriminant with numerical stability
            discriminant = b * b - 4.0 * a * cc
            discriminant = torch.clamp(discriminant, min=eps)  # Ensure non-negative
            
            # Solve for Zs using quadratic formula (take positive root)
            zs = (-b + torch.sqrt(discriminant)) / (2.0 * a + eps)
            
            # Compute 3D coordinates
            factor = zs + xi

            Xc, Yc, Zc = xu * factor * d, yu * factor * d, zs * d
            P_c = torch.stack([Xc, Yc, Zc, torch.ones_like(Xc)], dim=-1)
            # get distortion map 2d
            dist_coords_2d = torch.stack([Xc[...,0], Yc[...,0],Zc[...,0]], dim=-1)
            P_lidar = torch.matmul(cam2lidar, P_c.unsqueeze(-1)).squeeze(-1)
            coords3d = P_lidar[..., :3]
        else:
            coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
            coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
            coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
            intrinsics = []
            extrinsics = []
            for img_meta in img_metas:
                intrinsics.append(img_meta['intrinsics'])
                extrinsics.append(img_meta['extrinsics'])
            intrinsics = coords.new_tensor(np.asarray(intrinsics)) # (B, N, 4, 4)
            extrinsics = coords.new_tensor(np.asarray(extrinsics)) # (B, N, 4, 4)
            inv_intrinsics = torch.inverse(intrinsics).view(B, N, 1, 1, 1, 4, 4)
            cam2lidar = torch.inverse(extrinsics).view(B, N, 1, 1, 1, 4, 4)

            coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
            inv_intrinsics = inv_intrinsics.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
            cam2lidar = cam2lidar.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
            coords = torch.matmul(inv_intrinsics, coords)
            coords_2d = coords[..., 0, :, 0]  # [B, N, W, H, 4]
            dist_coords_2d = coords_2d[..., :3] / coords_2d[..., 2:3]  # [B, N, W, H, 2]
            coords3d = torch.matmul(cam2lidar, coords).squeeze(-1)[..., :3]
        # Convert Cartesian to Cylindrical coordinates for BEV
        rho = torch.sqrt(coords3d[..., 0]**2 + coords3d[..., 1]**2)
        phi = torch.atan2(coords3d[..., 1], coords3d[..., 0])
        z = coords3d[..., 2]
        coords3d = torch.stack([rho, phi, z], dim=-1)

        # Normalize cylindrical coordinates to be within [0, 1]
        max_rho = (self.position_range[3]**2 + self.position_range[4]**2)**0.5
        coords3d[..., 0:1] = coords3d[..., 0:1] / max_rho
        coords3d[..., 1:2] = (coords3d[..., 1:2] + math.pi) / (2 * math.pi)
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0) 
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), dist_coords_2d.permute(0, 1, 4, 3, 2), coords_mask
    def extract_distortion_params(self, dist_coords_2d):
        """Extract distortion parameters from 2D coordinates.
        Args:
            dist_coords_2d (Tensor): Distorted 2D coordinates of shape
                (B, N, 3, H, W), where B is batch size, N is number of cameras,
                H and W are height and width of the feature map.
        Returns:
            Tensor: Extracted distortion parameters of shape (B, N, 5, H, W).
        """
        B, N, _, H, W = dist_coords_2d.shape
        x = dist_coords_2d[:, :, 0, :, :]
        y = dist_coords_2d[:, :, 1, :, :]
        z = dist_coords_2d[:, :, 2, :, :]
        r_norm = torch.sqrt(x**2 + y**2 + z**2) + 1e-6  # Avoid division by zero
        rays = z / r_norm
        angles = torch.atan2(y, x)
        radial = torch.sqrt(x**2 + y**2) / r_norm
        # Reshape for FiLM input: [B*N, 1, H, W]
        rays = rays.view(B*N, 1, H, W)
        angles = angles.view(B*N, 1, H, W)
        radial = radial.view(B*N, 1, H, W)
        return rays, angles, radial
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is PETRHeadPolar4Nusc:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        x = mlvl_feats[0]
        batch_size, num_cams = x.size(0), x.size(1)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        x = self.input_proj(x.flatten(0,1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)

        if self.with_position:
            if self.with_polar_pe:
                coords_position_embeding, dist_coords_2d, _ = self.position_embeding(mlvl_feats, img_metas, masks)
            else:
                coords_position_embeding, dist_coords_2d , _= self.position_embeding(mlvl_feats, img_metas, masks)
            # Apply FiLM conditioning: condition image features with 3D position embeddings
            if self.with_film:
                x_modulated = self.film(x.flatten(0,1), coords_position_embeding.flatten(0,1))
                # # Extract distortion parameters from dist_coords_2d
                # rays, angles, radial = self.extract_distortion_params(dist_coords_2d)
                # x_modulated = self.film(
                #     features=x.flatten(0,1),  # [B*N, C, H, W]
                #     rays=rays,                # [B*N, 1, H, W]
                #     angles=angles,            # [B*N, 1, H, W]
                #     radial=radial           # [B*N, 1, H, W]
                # )
                x = x_modulated.view(x.size())

            if self.with_fpe:
                coords_position_embeding = self.fpe(coords_position_embeding.flatten(0,1), x.flatten(0,1)).view(x.size())
            pos_embed = coords_position_embeding
            if self.with_multiview:
                if self.with_polar_pe:
                    sin_embed = self.positional_encoding(masks, dist_coords_2d)
                else:
                    sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    if self.with_polar_pe:
                        xy_embed = self.positional_encoding(masks[:, i, :, :], dist_coords_2d[:, i, :, :, :])
                    else:
                        xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.reference_points.weight
        
        # reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches)
        # outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, attn_mask, self.reg_branches)
        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        max_rho = (self.pc_range[3]**2 + self.pc_range[4]**2)**0.5
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            # Un-normalize rho, theta, z
            tmp[..., 0:1] = tmp[..., 0:1] * max_rho
            tmp[..., 1:2] = (tmp[..., 1:2] * (2 * math.pi)) - math.pi
            tmp[..., 2:3] = tmp[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            tmp_ = tmp.clone()
            # convert to cartesian coordinates
            tmp_[..., 0:1] = tmp[..., 0:1] * torch.cos(tmp[..., 1:2])
            tmp_[..., 1:2] = tmp[..., 0:1] * torch.sin(tmp[..., 1:2])
            outputs_coord = tmp_
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        return outs
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt
    
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        bbox_weights[:, 6:8] = 0  ###dn alaways reduce the mAOE, which is useless when training for a long time.
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
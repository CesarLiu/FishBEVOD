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
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core import (CameraInstance3DBoxes,LiDARInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.fisheye_coords import FisheyeCoordAug


@DETECTORS.register_module()
class Petr3DFilm(MVXTwoStageDetector):
    """Petr3DFilm."""

    def __init__(self,
                 use_grid_mask=False,
                 use_cam_conv=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 fisheye_aug=None):
        super(Petr3DFilm, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.use_cam_conv = use_cam_conv
        
        # Fisheye augmentation module with new API
        self.fisheye_enabled = False
        self.fisheye_add_rays = True
        self.fisheye_add_angles = False
        
        if fisheye_aug is not None and isinstance(fisheye_aug, dict):
            self.fisheye_enabled = fisheye_aug.get('enabled', False)
            self.fisheye_add_rays = fisheye_aug.get('add_rays', True)
            self.fisheye_add_angles = fisheye_aug.get('add_angles', False)
            self.fisheye_add_radial = fisheye_aug.get('add_radial', False)
            
        if self.fisheye_enabled:
            # Create augmentation module with new parameters
            self.fisheye_aug = FisheyeCoordAug(
                add_rays=self.fisheye_add_rays,
                add_angles=self.fisheye_add_angles,
                add_radial=self.fisheye_add_radial  # Note: parameter name is 'add_radial' in implementation
            )

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        # print(f'original image size is {img.size()}')
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            # Build fisheye-aware geometry features if enabled
            dist_feats = None
            if getattr(self, 'fisheye_enabled', False):
                # Build BN-aligned intrinsics (3x3) and distortion (5)
                if img.dim() == 4:
                    # (BN, C, H, W), deduce batch and views from metas
                    BN, C, H, W = img.size()
                    B = len(img_metas)
                    N = len(img_metas[0]['intrinsics']) if B > 0 else 0
                else:
                    # already have B,N from above branch
                    BN, C, H, W = img.size()
                    
                K_list = []
                D_list = []
                for b, meta in enumerate(img_metas):
                    intrs = meta.get('intrinsics', [])
                    dists = meta.get('dist_coeffs', [])
                    for v in range(len(intrs)):
                        K4 = intrs[v]
                        # Handle numpy or list
                        K3 = torch.as_tensor(K4, dtype=img.dtype)[:3, :3]
                        K_list.append(K3)
                        if isinstance(dists, (list, tuple)) and len(dists) > v and dists[v] is not None:
                            D = torch.as_tensor(dists[v], dtype=img.dtype)
                        else:
                            D = torch.zeros(5, dtype=img.dtype)
                        # Ensure shape (5,) -> [k1,k2,p1,p2,xi]
                        if D.numel() == 4:
                            # If xi missing, append 0
                            D = torch.cat([D, D.new_zeros(1)], dim=0)
                        D_list.append(D)
                        
                if K_list:
                    K = torch.stack(K_list, dim=0).to(img.device)
                    dist = torch.stack(D_list, dim=0).to(img.device)
                    if K.shape[0] != BN:
                        reps = int((BN + K.shape[0] - 1) // max(K.shape[0], 1)) if K.shape[0] > 0 else 1
                        K = K.repeat(reps, 1, 1)[:BN]
                        dist = dist.repeat(reps, 1)[:BN]
                    # Compute fisheye features as dictionary
                    dist_feats = self.fisheye_aug(img, K, dist)  # Returns dict with 'rays', 'angles', 'lambda'
            # Backbone forward; pass dist_feats to FiLM-enabled backbones
            try:
                # Normalize and concatenate dist_feats to img
                if dist_feats is not None and self.use_cam_conv:
                    rays = dist_feats.get('rays', None)
                    angles = dist_feats.get('angles', None)
                    radial = dist_feats.get('radial', None)
                    norm_feats = []
                    if rays is not None:
                        rays = torch.clamp(rays, -0.5, 1.0) / 1.5
                        norm_feats.append(rays)
                    if angles is not None:
                        angles = torch.clamp(angles, -1.6, 1.6) / 1.6
                        norm_feats.append(angles)
                    if radial is not None:
                        radial = torch.clamp(radial, 0, 1.0)
                        norm_feats.append(radial)
                    if norm_feats:
                        # Concatenate along channel dimension
                        norm_feats = torch.cat(norm_feats, dim=1)
                        img = torch.cat([img, norm_feats], dim=1)
                img_feats = self.img_backbone(img, dist_feats=dist_feats)
            except TypeError:
                img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            try:
                img_feats = self.img_neck(img_feats, dist_feats=dist_feats)
            except TypeError:
                img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
  
    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    
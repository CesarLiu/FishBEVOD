_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = ['car', 'truck', 'trailer', 'bus', 
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier','trafficsign'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Petr3DROI',
    use_grid_mask=True,
    img_backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(2, 3), # tiny: 384,768 large: 768,1536 [96, 192, 384, 768]
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
    ),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[384, 768],
        out_channels=256,
        num_outs=3),
    img_roi_head=dict(
        type='ROIHead',
        num_classes=10,
        in_channels=256,
        stride=16, # 4 for swin large, 8 for swin tiny
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=2.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0), #5.0
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),# 2.0
        loss_centers2d=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
        assigner2d=dict(
            type='HungarianAssigner2D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            centers2d_cost=dict(type='BBox3DL1Cost', weight=1.0)))
        ),
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        with_fpe = True,
        with_distortion=False,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        # code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention', 
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
            positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range),)))


dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')


ida_aug_conf = {
        "resize_lim": (0.94, 1.25),
        "final_dim": (376, 1408),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 376,
        "W": 1408,
        "rand_flip": True,
    }
train_pipeline = [
    dict(
        type='LoadMixedCameraImageFromFiles',
        to_float32=True,
        num_pinhole=2,
        num_fisheye=2,
        with_2d=True,
        fisheye_crop_regions=[
            [14, 600, 1364, 960],  # Crop region for left fisheye camera
            [50, 600, 1400, 960]   # Crop region for right fisheye camera
        ]
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths'],
             meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'gt_bboxes_3d','gt_labels_3d','dist_coeffs'))
]
test_pipeline = [
    dict(
        type='LoadMixedCameraImageFromFiles',
        to_float32=True,
        num_pinhole=2,
        num_fisheye=2,
        fisheye_crop_regions=[
            [14, 600, 1364, 960],  # Crop region for left fisheye camera
            [50, 600, 1400, 960]   # Crop region for right fisheye camera
        ]
    ),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1408, 376),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'dist_coeffs'))
        ])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti360_2d_temporal_infos_train_11k.pkl', #'kitti360_2d_temporal_infos_train_11k.pkl'
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        with_2d_labels=True,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, with_2d_labels=True,ann_file=data_root + 'kitti360_2d_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, with_2d_labels=True,ann_file=data_root + 'kitti360_2d_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    # shuffler_sampler=dict(type='DistributedGroupSampler'),
    # nonshuffler_sampler=dict(type='DistributedSampler')
    )

optimizer = dict(
    type='AdamW', 
    lr=3e-4, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

total_epochs = 24
evaluation = dict(interval=4, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# load_from='ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from='work_dirs/focal_petrv1_swint_flash_800_24e/latest.pth'
# mAP: 0.4223                                                                                                                                                                                                                        
# mATE: 0.6575
# mASE: 0.2617
# mAOE: 0.4600
# mAVE: 0.7613
# mAAE: 0.2034
# NDS: 0.4767
# Eval time: 129.6s

# Per-class results:                                                                                                                                                                                                                                     Scenes/
# Object Class            AP      ATE     ASE     AOE     AVE     AAE                                                                                                                                                                                    , pts_b
# car                     0.605   0.485   0.147   0.072   0.826   0.213                                                                                                                                                                                  ck_AP_d
# truck                   0.375   0.694   0.196   0.096   0.798   0.241                                                                                                                                                                                  /constr
# bus                     0.468   0.693   0.194   0.105   1.667   0.310                                                                                                                                                                                  vehicle
# trailer                 0.243   0.945   0.232   0.566   0.572   0.156                                                                                                                                                                                  8, pts_
# construction_vehicle    0.091   0.981   0.449   1.326   0.145   0.358                                                                                                                                                                                  4, pts_
# pedestrian              0.492   0.629   0.286   0.580   0.503   0.215                                                                                                                                                                                  0: 0.34
# motorcycle              0.398   0.642   0.253   0.552   1.203   0.121                                                                                                                                                                                  /traile
# bicycle                 0.394   0.556   0.256   0.717   0.377   0.013                                                                                                                                                                                  4886, p
# traffic_cone            0.599   0.462   0.316   nan     nan     nan                                                                                                                                                                                    le_AP_d
# barrier                 0.557   0.489   0.287   0.125   nan     nan

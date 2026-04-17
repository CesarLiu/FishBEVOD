# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import mmcv
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample

import argparse


cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
#  'CAM_BACK',
 'CAM_BACK_LEFT'
#  'CAM_FRONT_LEFT'
]

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams


def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = 'render.png',
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ ,_= nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('bbox in cams:', select_cams)
    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic,_ = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic,_ = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)



def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
        if cs_record['distortion_coefficients'] == "":
            dist_coeffs = None
        else:
            dist_coeffs = np.array(cs_record['distortion_coefficients'])
    else:
        cam_intrinsic = None
        imsize = None
        dist_coeffs = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level,dist_coeffs=dist_coeffs):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic, dist_coeffs



def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
        if cs_record['distortion_coefficients'] == "":
            dist_coeffs = None
        else:
            dist_coeffs = np.array(cs_record['distortion_coefficients'])
    else:
        cam_intrinsic = None
        imsize = None
        dist_coeffs = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level,dist_coeffs=dist_coeffs):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic, dist_coeffs




def lidiar_render(sample_token, data,out_path=None):
    bbox_gt_list = []
    bbox_pred_list = []
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        try:
            bbox_gt_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=category_to_detection_name(content['category_name']),
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))
        except:
            pass

    bbox_anns = data['results'][sample_token]
    for content in bbox_anns:
        bbox_pred_list.append(DetectionBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name']))
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    print('green is ground truth')
    print('get groundtruthnumber:', len(bbox_gt_list))
    print('blue is the predited result')
    print('get predicted number:', len(bbox_pred_list))
    visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, savepath=out_path+'_bev')


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def render_sample_data(
        sample_toekn: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
        with_distortion: bool = True,
        det_score_thresh: float = 0.2
      ) -> None:
    """
    Render sample data with LiDAR BEV on top for predictions, cameras only for ground truth.
    """
    sample = nusc.get('sample', sample_toekn)
    cams = [
        # 'CAM_FRONT',
        # 'CAM_FRONT_RIGHT', 
        'CAM_BACK_RIGHT',
        'CAM_BACK_LEFT'
        ]
    
    # Create custom layout using GridSpec for better control
    from matplotlib import gridspec
    
    # Create separate figures with different layouts
    # Predictions: LiDAR BEV on top + 4 camera views
    fig_pred = plt.figure(figsize=(20, 18))
    gs_pred = gridspec.GridSpec(2, 2, height_ratios=[3.5, 3.5], width_ratios=[1, 1], 
                               hspace=0.08, wspace=0.05)
    
    # Ground truth: Only 4 camera views  
    # fig_gt = plt.figure(figsize=(20, 12))
    # gs_gt = gridspec.GridSpec(2, 2, height_ratios=[1, 3.5], width_ratios=[1, 1], 
    #                          hspace=0.05, wspace=0.05)
    
    # Create axes for predictions (with BEV on top)
    ax_pred_bev = fig_pred.add_subplot(gs_pred[0, :])  # BEV spans full width
    ax_pred = [fig_pred.add_subplot(gs_pred[1, 0]), fig_pred.add_subplot(gs_pred[1, 1])]
    
    # Create axes for ground truth (cameras only)
    # ax_gt = [fig_gt.add_subplot(gs_gt[0, 0]), fig_gt.add_subplot(gs_gt[0, 1]),
    #          fig_gt.add_subplot(gs_gt[1, 0]), fig_gt.add_subplot(gs_gt[1, 1])]
    
    # Render LiDAR BEV for predictions
    lidar_token = sample['data']['LIDAR_TOP']
    
    # Get sensor and pose records for coordinate transformation
    sd_record = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    
    # Get predicted boxes and convert to DetectionBox format
    pred_detection_boxes = []
    for record in pred_data['results'][sample_toekn]:
        if record['detection_score'] > det_score_thresh:
            pred_detection_boxes.append(DetectionBox(
                sample_token=record['sample_token'],
                translation=tuple(record['translation']),
                size=tuple(record['size']),
                rotation=tuple(record['rotation']),
                velocity=tuple(record['velocity']),
                detection_name=record['detection_name'],
                detection_score=record['detection_score']
            ))
    
    # Get GT boxes and convert to DetectionBox format
    gt_detection_boxes = []
    anns = nusc.get('sample', sample_toekn)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        try:
            gt_name = category_to_detection_name(content['category_name'])
            if gt_name == 'construction_vehicle':
                continue
            gt_detection_boxes.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                detection_name=category_to_detection_name(content['category_name']),
                detection_score=1.0
            ))
        except:
            pass
    
    # Transform boxes to sensor coordinates using nuscenes utility
    from nuscenes.eval.common.utils import boxes_to_sensor
    boxes_pred_sensor = boxes_to_sensor(pred_detection_boxes, pose_record, cs_record)
    boxes_gt_sensor = boxes_to_sensor(gt_detection_boxes, pose_record, cs_record)
    
    # Add scores to predicted boxes
    for box_pred, box_pred_global in zip(boxes_pred_sensor, pred_detection_boxes):
        box_pred.score = box_pred_global.detection_score
    
    # Get point cloud and render
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=1)
    
    # Render point cloud
    from nuscenes.utils.geometry_utils import view_points
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    eval_range = 50
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax_pred_bev.scatter(points[0, :], points[1, :], c=colors, s=0.2)
    
    # Show ego vehicle
    ax_pred_bev.plot(0, 0, 'x', color='black', markersize=10)
    
    # Render GT boxes
    for box in boxes_gt_sensor:
        box.render(ax_pred_bev, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)
    
    # Render predicted boxes
    conf_th = det_score_thresh
    for box in boxes_pred_sensor:
        if hasattr(box, 'score') and box.score >= conf_th:
            box.render(ax_pred_bev, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)
    
    # Set up the BEV plot
    axes_limit = eval_range + 3
    ax_pred_bev.set_xlim(-axes_limit*1.8, axes_limit*1.8)
    ax_pred_bev.set_ylim(-axes_limit*1.2, axes_limit*1.2)
    ax_pred_bev.set_title('BEV: Predictions (Blue) vs Ground Truth (Green)', fontsize=12, pad=10)
    ax_pred_bev.axis('off')
    ax_pred_bev.set_aspect('equal')
    
    # Render camera views
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]
        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            assert False
        elif sensor_modality == 'camera':
            # Load boxes and image.
            # print('Visualize on Camera with original prediction number:', len(pred_data['results'][sample_toekn]))
            boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                         name=record['detection_name'], token='predicted') for record in
                     pred_data['results'][sample_toekn] if record['detection_score'] > det_score_thresh and record['detection_name']!='construction_vehicle']

            # print('after filter prediction number:', len(boxes))
            data_path, boxes_pred, camera_intrinsic,dist_coeffs = get_predicted_data(sample_data_token,
                                                                         box_vis_level=box_vis_level, pred_anns=boxes)
            if not with_distortion:
                dist_coeffs = None
            # _, boxes_gt, _,_ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
            
            data = Image.open(data_path)

            # Show image on both figures
            ax_pred[ind].imshow(data)
            # ax_gt[ind].imshow(data)

            # Show boxes
            if with_anns:
                # Predictions
                for box in boxes_pred:
                    c = np.array(get_color(box.name)) / 255.0
                    box.render(ax_pred[ind], view=camera_intrinsic, normalize=True, colors=(c, c, c),dist_coeffs=dist_coeffs)
                
                # # Ground truth
                # for box in boxes_gt:
                #     c = np.array(get_color(box.name)) / 255.0
                #     box.render(ax_gt[ind], view=camera_intrinsic, normalize=True, colors=(c, c, c),dist_coeffs=dist_coeffs)

            # Limit visible range for both figures
            ax_pred[ind].set_xlim(0, data.size[0])
            ax_pred[ind].set_ylim(data.size[1], 0)
            # ax_gt[ind].set_xlim(0, data.size[0])
            # ax_gt[ind].set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        # Set titles and formatting for predictions
        ax_pred[ind].axis('off')
        cam_name_mapping = {
                            'CAM_BACK_RIGHT': 'CAM_Fish_Right', 
                            'CAM_BACK_LEFT': 'CAM_Fish_Left',
                        }
        ax_pred[ind].set_title('PRED: {}'.format(cam_name_mapping.get(sd_record['channel'], sd_record['channel'])), fontsize=10, pad=5)
        ax_pred[ind].set_aspect('equal')

        # Set titles and formatting for ground truth
        # ax_gt[ind].axis('off')
        # ax_gt[ind].set_title('GT: {}'.format(sd_record['channel']), fontsize=10, pad=5)
        # ax_gt[ind].set_aspect('equal')
    
    # Adjust layout to make it tight
    fig_pred.tight_layout(pad=0.5)
    # fig_gt.tight_layout(pad=0.5)
    
    # Save both figures if output path is specified
    if out_path is not None:
        fig_pred.savefig(out_path + '_camera_pred', bbox_inches='tight', pad_inches=0.1, dpi=200)
        # fig_gt.savefig(out_path + '_camera_gt', bbox_inches='tight', pad_inches=0.1, dpi=200)
    
    if verbose:
        plt.show()
    
    # Close both figures
    plt.close(fig_pred)
    # plt.close(fig_gt)
def create_video_from_images(image_paths, output_path, fps=2):
    """
    Create a video from a list of image paths.
    """
    if not image_paths:
        print("No images found for video creation")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    height, width, layers = first_image.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video with {len(image_paths)} frames at {fps} FPS...")
    
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"Warning: Could not read image {image_path}")
    
    video_writer.release()
    print(f"Video created successfully: {output_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize NuScenes detection results.")
    parser.add_argument('--result_json', type=str, required=True, help='Path to detection result JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output figures')
    parser.add_argument('--step', type=int, default=10, help='Step size for visualization')
    parser.add_argument('--nusc_root', type=str, default='./data/nuscenes', help='NuScenes data root')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='NuScenes version')
    parser.add_argument('--max_samples', type=int, default=100, help='Number of samples to visualize')
    parser.add_argument('--with_distortion', action='store_true', help='Whether to render on distorted images or rectified images')
    parser.add_argument('--det_score_thresh', type=float, default=0.2, help='Threshold for detection score to visualize')
    parser.add_argument('--create_video', action='store_true', help='Create video from saved images')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second for video')
    args = parser.parse_args()
    
    import os
    import cv2
    from pathlib import Path
    
    os.makedirs(args.output_dir, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)
    bevformer_results = mmcv.load(args.result_json)
    sample_token_list = list(bevformer_results['results'].keys())
    print(f"Total samples in results: {len(sample_token_list)}")
    
    # Generate images and keep track of the order
    processed_tokens = []
    for idx in range(0, min(args.max_samples, len(sample_token_list)), args.step):
        token = sample_token_list[idx]
        out_path = os.path.join(args.output_dir, token)
        render_sample_data(token, pred_data=bevformer_results, out_path=out_path,
                          with_distortion=args.with_distortion, det_score_thresh=args.det_score_thresh)
        processed_tokens.append(token)
    
    # Create videos if requested
    if args.create_video:
        print("Creating videos from saved images...")
        
        # Build image paths in the correct order based on processed tokens
        pred_images = []
        gt_images = []
        
        for token in processed_tokens:
            pred_path = os.path.join(args.output_dir, f"{token}_camera_pred.png")
            gt_path = os.path.join(args.output_dir, f"{token}_camera_gt.png")
            
            if os.path.exists(pred_path):
                pred_images.append(pred_path)
            if os.path.exists(gt_path):
                gt_images.append(gt_path)
        
        if pred_images:
            # Create prediction video
            pred_video_path = os.path.join(args.output_dir, 'predictions_video_20fps.mp4')
            create_video_from_images(pred_images, pred_video_path, args.fps)
            print(f"Prediction video saved to: {pred_video_path}")
        
        # if gt_images:
        #     # Create ground truth video
        #     gt_video_path = os.path.join(args.output_dir, 'ground_truth_video_20fps.mp4')
        #     create_video_from_images(gt_images, gt_video_path, args.fps)
        #     print(f"Ground truth video saved to: {gt_video_path}")



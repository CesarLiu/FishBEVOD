# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
import torch
@PIPELINES.register_module()
class LoadMapsFromFiles(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(256,3*256)
        results['map_shape']=maps.shape
        results['maps']=maps
        return results



@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                sweeps_num=5,
                to_float32=False, 
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                sweep_range=[3,27],
                sweeps_id = None,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                prob=1.0,
                ):

        self.sweeps_num = sweeps_num    
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['sweeps']))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
                
            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]

                print(f"Processing sweep {sweep_idx}: Sensors available: {list(sweep.keys())}")
                print([sweep['type']])
                print([sweep[sensor] for sensor in self.sensors])


                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list  

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMapsFromFiles_flattenf200f3(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        # maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(3,200*200)
        maps[maps>=0.5]=1
        maps[maps<0.5]=0
        maps=1-maps
        results['map_shape']=maps.shape
        results['maps']=maps
        
        return results

@PIPELINES.register_module()
class LoadMixedCameraImageFromFiles(object):
    """Load multi-view images from files with support for different camera types.
    
    Handles pinhole and fisheye cameras with different sizes.
    Can crop fisheye images to specified regions and adjusts projection matrices accordingly.
    Resizes fisheye images to match pinhole image dimensions.

    Args:
        to_float32 (bool): Whether to convert the img to float32. Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        fisheye_crop_regions (list): List of [x0,y0,x1,y1] crop regions for fisheye images.
                                    Should match the number of fisheye cameras.
        num_pinhole (int): Number of pinhole cameras/images to load. Defaults to 2.
        num_fisheye (int): Number of fisheye cameras/images to load. Defaults to 2.
    """

    def __init__(self, 
                 to_float32=False, 
                 color_type='unchanged',
                 fisheye_crop_regions=None,
                 num_pinhole=2,
                 num_fisheye=2,
                 drop_strategy='zeros',
                 drop_img_indices=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.fisheye_crop_regions = fisheye_crop_regions
        self.num_pinhole = num_pinhole
        self.num_fisheye = num_fisheye
        self.drop_img_indices = drop_img_indices or []
        self.drop_strategy = drop_strategy or 'zeros'
        
        # Validate fisheye crop regions if provided
        if self.fisheye_crop_regions:
            assert len(self.fisheye_crop_regions) == self.num_fisheye, \
                f"Number of crop regions ({len(self.fisheye_crop_regions)}) " \
                f"must match number of fisheye cameras ({self.num_fisheye})"
        # Validate drop indices
        # Validate inputs
        total_cameras = self.num_pinhole + self.num_fisheye
        for idx in self.drop_img_indices:
            assert 0 <= idx < total_cameras, \
                f"Drop index {idx} out of range [0, {total_cameras-1}]"
    def _create_mock_image(self, target_shape, strategy='zeros'):
        """Create a mock image based on the specified strategy.
        
        Args:
            target_shape (tuple): Target image shape (H, W, C)
            strategy (str): Mock strategy to use
            
        Returns:
            np.ndarray: Mock image
        """
        if strategy == 'zeros':
            mock_img = np.zeros(target_shape, dtype=np.uint8)
        elif strategy == 'noise':
            # Generate random noise in [0, 255] range
            noise = np.random.normal(127.5, self.noise_std * 255, target_shape)
            mock_img = np.clip(noise, 0, 255).astype(np.uint8)
        elif strategy == 'mean':
            # Create a mean gray image
            mean_value = 127  # Middle gray
            mock_img = np.full(target_shape, mean_value, dtype=np.uint8)
        else:
            # For 'duplicate' strategy, this will be handled separately
            mock_img = np.zeros(target_shape, dtype=np.uint8)
            
        return mock_img
    
    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames and cam_intrinsics.

        Returns:
            dict: The result dict containing the multi-view image data.
        """
        filename = results['img_filename']
        assert len(filename) == self.num_pinhole + self.num_fisheye, \
            f"Expected {self.num_pinhole + self.num_fisheye} filenames, got {len(filename)}"
        
        pinhole_imgs = []
        fisheye_imgs = []
        
        # Load pinhole images first (no cropping needed)
        for i in range(self.num_pinhole):
            img = mmcv.imread(filename[i], self.color_type)
            pinhole_imgs.append(img)
        
        # Get target size from first pinhole image
        target_height, target_width = pinhole_imgs[0].shape[:2]
        
        # Load and crop fisheye images
        for i in range(self.num_fisheye):
            img = mmcv.imread(filename[self.num_pinhole + i], self.color_type)
            
            # Apply cropping if crop regions are provided
            if self.fisheye_crop_regions:
                crop_region = self.fisheye_crop_regions[i]
                x0, y0, x1, y1 = crop_region
                img = img[y0:y1, x0:x1]
                
                # Get the cropped image dimensions
                crop_height, crop_width = img.shape[:2]
                
                # Calculate scale factors for resizing
                scale_x = target_width / crop_width
                scale_y = target_height / crop_height
                
                # Adjust camera intrinsics before resizing if they exist
                cam_keys = ['intrinsics']
                for cam_key in cam_keys:
                    if cam_key in results:
                        # Adjust principal point for cropping
                        results[cam_key][self.num_pinhole + i][0, 2] -= x0  # cx adjustment
                        results[cam_key][self.num_pinhole + i][1, 2] -= y0  # cy adjustment
                        
                        # Scale intrinsics for resize
                        # fx and fy are scaled by scale factors
                        results[cam_key][self.num_pinhole + i][0, 0] *= scale_x  # fx adjustment
                        results[cam_key][self.num_pinhole + i][1, 1] *= scale_y  # fy adjustment
                        # cx and cy are also scaled by scale factors
                        results[cam_key][self.num_pinhole + i][0, 2] *= scale_x  # cx adjustment
                        results[cam_key][self.num_pinhole + i][1, 2] *= scale_y  # cy adjustment
                
                # Resize the cropped image to match pinhole image dimensions
                img = mmcv.imresize(img, (target_width, target_height), return_scale=False)
            
            fisheye_imgs.append(img)
        
        # Combine all images
        all_imgs = pinhole_imgs + fisheye_imgs
        # Get sample shape from first valid image
        sample_shape = None
        if sample_shape is None:
            sample_shape = (target_height, target_width, 3)
        
        # Fallback shape if all images are None
        if sample_shape is None:
            sample_shape = (target_height, target_width, 3)
        # Handle dropped images
        for i in self.drop_img_indices:
            if i < len(all_imgs):
                all_imgs[i] = self._create_mock_image(sample_shape, self.drop_strategy)
        
        # Convert to float32 if needed
        if self.to_float32:
            all_imgs = [img.astype(np.float32) for img in all_imgs]
        
        results['filename'] = filename
        results['img'] = all_imgs
        
        # Store individual image shapes (now all should be the same after resizing)
        sample_img_shape = all_imgs[0].shape  # Shape of a single image
        img_shapes = sample_img_shape[:2] + (sample_img_shape[2], len(all_imgs))
        results['img_shape'] = img_shapes
        results['ori_shape'] = img_shapes
        
        # Set initial values for default meta_keys
        results['pad_shape'] = img_shapes
        results['scale_factor'] = 1.0
        
        # Handle normalization config (all images should have same channels now)
        num_channels = 1 if len(img_shapes) < 3 else img_shapes[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f"num_pinhole={self.num_pinhole}, "
        repr_str += f"num_fisheye={self.num_fisheye}, "
        repr_str += f"fisheye_crop_regions={self.fisheye_crop_regions})"
        return repr_str
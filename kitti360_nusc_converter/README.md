# kitti360_nusc_converter

Convert KITTI-360 into a nuScenes-style dataset layout for training, visualization, and devkit-based inspection.

## Goal

This repository builds a `vkitti360-trainval` split from KITTI-360 and a smaller `vkitti360-mini` split for debugging and experiments. The current recommended pipeline is the unified converter in `convert_kitti360_to_nuscenes.py`.

## Current Progress

### Full trainval conversion

Current `vkitti360-trainval` output after the fixes:

- 300 scenes
- 68,893 samples
- 68,893 ego poses
- 612,645 sample_data records
- 2,276,698 sample annotations
- 17,700 instances
- 40 categories

Validation status:

- Referential integrity checks pass.
- nuScenes devkit loading works when map masks are available under `<dataroot>/maps/`.
- 257 samples have no annotations.

### About the 257 samples without annotations

The remaining unannotated samples are localized, not spread across the dataset:

- All 257 belong to `scene-0093`.
- `scene-0093` corresponds to sequence `2013_05_28_drive_0002_sync`, frames `15886-16223`.

This is typically acceptable for nuScenes-style datasets. Empty samples act as negatives and do not break dataset integrity. They only matter if a downstream training or evaluation script assumes every sample must contain at least one box.

### Mini dataset

The generated `vkitti360-mini` dataset was tested successfully:

- 6 scenes
- 1,383 samples
- 12,301 sample_data records
- 51,113 annotations
- 999 instances

Both referential checks and nuScenes devkit loading passed for the mini dataset.

## Recommended Pipeline

Use the unified converter rather than the older one-script-per-table pipeline.

### Phase 1: Static tables

Generate:

- `sensor.json`
- `calibrated_sensor.json`
- `log.json`
- `map.json`
- `category.json`
- `attribute.json`
- `visibility.json`

Key details:

- Camera and lidar extrinsics are derived from KITTI-360 calibration files.
- Fisheye cameras include distortion coefficients.
- Categories preserve native KITTI-360 granularity where possible.

### Phase 2: Scenes, samples, ego poses, sample data

Generate:

- `scene.json`
- `split.json`
- `sample.json`
- `ego_pose.json`
- `sample_data.json`

Key details:

- Scene windows come from KITTI-360 semantic train/val split files.
- Overlapping windows are deduplicated so each frame belongs to exactly one scene.
- Samples are keyed by velodyne frames.
- Cameras are written as keyframes.
- SICK scans are assigned by timestamp window and the closest SICK scan becomes the SICK keyframe for that sample.

### Phase 3: Annotations

Generate:

- `instance.json`
- `sample_annotation.json`

Key details:

- Static and dynamic boxes are both validated by lidar point-in-box counts.
- Ground filtering is done in sensor-local coordinates.
- Dynamic objects are indexed by frame for faster lookup.
- Distance gating is used before expensive point-in-box evaluation.

### Phase 4: Validation

Run:

- referential integrity checks over all JSON tables
- optional nuScenes devkit loading

## Frame Assignment Method

The current frame assignment policy is important because KITTI-360 train/val semantic windows can overlap.

### Scene window assignment

The converter reads split windows from:

- `data_3d_semantics/train/2013_05_28_drive_train.txt`
- `data_3d_semantics/train/2013_05_28_drive_val.txt`

Then it applies `_deduplicate_windows()`:

1. Group windows by sequence.
2. Sort windows by start frame.
3. If a window overlaps the previous one, clip its start to `prev_end + 1`.
4. Drop the window if clipping makes it empty.

Result: each frame belongs to at most one scene.

### Sample assignment

Within each deduplicated scene window:

1. A frame is kept only if the velodyne file exists.
2. The frame must have a valid pose from either `poses.txt` or OXTS fallback.
3. The sample timestamp is the velodyne timestamp.
4. The ego pose token is set 1:1 with the sample token.

This makes velodyne the temporal backbone of the dataset.

### Pose assignment

Pose loading uses a merged strategy:

1. Load OXTS poses.
2. Convert OXTS poses with `convertOxtsToPose()`.
3. Apply `postprocessPoses()` to flip from NED to FLU.
4. Override any available frames with KITTI-360 `poses.txt` SLAM-corrected poses.

This fixes the early-frame visualization bug where OXTS fallback poses were previously left in the wrong coordinate convention.

### SICK frame assignment

SICK is not treated as a single sweep-only stream.

For each velodyne sample:

1. Compute a time window bounded by the midpoints to the previous and next velodyne timestamps.
2. Assign every SICK scan in that window to the current sample.
3. Mark the closest SICK timestamp in that window as `is_key_frame = True`.
4. Keep all other SICK scans in the window as non-keyframe sweeps.

This reproduces the behavior used in the original multi-step pipeline while keeping the unified output valid for nuScenes-style access patterns.

## Scripts

### Main converter

Path:

- `nusc_converter/convert_kitti360_to_nuscenes.py`

Purpose:

- Build a full `vkitti360-trainval` dataset from raw KITTI-360.

Example:

```bash
python convert_kitti360_to_nuscenes.py \
	--kitti-root kitti360 \
	--output-dir kitti360/vkitti360-trainval \
	--version vkitti360-trainval
```

Main options:

- `--kitti-root`: KITTI-360 dataset root.
- `--output-dir`: output directory for the generated version.
- `--version`: version name used by the nuScenes devkit.
- `--skip-devkit`: skip nuScenes devkit validation.

### Mini dataset builder

Path:

- `kitti360/create_mini.py`

Purpose:

- Create a small nuScenes-style subset from an existing `vkitti360-trainval` output.

What it does:

1. Select scenes by name.
2. Filter `scene`, `log`, `map`, `sample`, `ego_pose`, `sample_data`, `sample_annotation`, `instance`, `category`, and `split`.
3. Copy shared global tables: `sensor`, `calibrated_sensor`, `attribute`, `visibility`.
4. Rebuild annotation and sample-data linked-list boundaries inside the mini split.
5. Rebuild instance first/last annotation pointers after filtering.
6. Ensure map masks exist under `<dataroot>/maps/`.
7. Validate the generated mini dataset.

Example:

```bash
python create_mini.py \
	--source-dir kitti360/vkitti360-trainval \
	--target-dir kitti360/vkitti360-mini \
	--overwrite
```

Custom scene selection:

```bash
python create_mini.py \
	--source-dir kitti360/vkitti360-trainval \
	--target-dir kitti360/vkitti360-mini-custom \
	--scene scene-0014 \
	--scene scene-0203 \
	--scene scene-0256 \
	--overwrite
```

### Older helper scripts

The repository still contains older one-table-at-a-time scripts in `nusc_converter/`, but they are not the recommended path anymore. They are useful mainly for debugging or archaeology.

## Dataset Layout Notes

Important KITTI-360 inputs used by the unified converter:

- `calibration/`
- `data_2d_raw/`
- `data_3d_raw/`
- `data_3d_bboxes/train/`
- `data_3d_semantics/train/`
- `data_poses/`

Important generated outputs:

- `vkitti360-trainval/`
- `vkitti360-mini/`

## Map Files and nuScenes Devkit

nuScenes expects map masks at:

- `<dataroot>/maps/<token>.png`


If map masks are missing from `<dataroot>/maps/`, the devkit fails during load. The mini builder handles this explicitly. For full trainval, make sure the dataroot-level `maps/` directory exists when validating with the nuScenes devkit.

## Useful References

1. KITTI-360 official scripts: https://github.com/autonomousvision/kitti360Scripts/tree/master

2. nuScenes tutorial: https://www.nuscenes.org/tutorials/nuscenes_tutorial.html

## Acknowledgements

This repository reuses and adapts parts of the official `kitti360Scripts` project, especially for pose loading and OXTS-to-pose conversion logic.

- Upstream project: `kitti360/kitti360Scripts`
- Upstream repository: https://github.com/autonomousvision/kitti360Scripts/tree/master
- Upstream license: `kitti360Scripts/LICENSE`

Thanks to the Autonomous Vision Group for releasing the original KITTI-360 scripts.

## License

This repository is licensed under the MIT License. See `LICENSE`.

Third-party code copied or adapted from `kitti360Scripts` remains subject to the original upstream MIT license notice included in `kitti360Scripts/LICENSE`.

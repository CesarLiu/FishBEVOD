# FishBEVOD: Benchmarking Multi-View BEV Object Detection with Mixed Pinhole and Fisheye Cameras

[![arXiv](https://img.shields.io/badge/arXiv-2603.27818-b31b1b.svg)](https://arxiv.org/abs/2603.27818)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Benchmarking Multi-View BEV Object Detection with Mixed Pinhole and Fisheye Cameras**  
> Xiangzhong Liu, Hao Shen  
> fortiss GmbH, Munich, Germany  
> ICRA 2026

## Abstract

Modern autonomous driving systems increasingly rely on mixed camera configurations with pinhole and fisheye cameras for full-view perception. However, Bird's-Eye View (BEV) 3D object detection models are predominantly designed for pinhole cameras, leading to performance degradation under fisheye distortion. To bridge this gap, we introduce a multi-view BEV detection benchmark with mixed cameras by converting KITTI-360 into nuScenes format. Our study encompasses three adaptations: rectification for zero-shot evaluation and fine-tuning of nuScenes-trained models, distortion-aware view transformation modules (VTMs) via the MEI camera model, and polar coordinate representations to better align with radial distortion. We systematically evaluate three representative BEV architectures—BEVFormer, BEVDet and PETR—across these strategies. We demonstrate that projection-free architectures are inherently more robust and effective against fisheye distortion. This work establishes the **first real-data 3D detection benchmark with fisheye and pinhole images** and provides systematic adaptation and practical guidelines for designing robust and cost-effective 3D perception systems.

## Repository Structure

```
FishBEVOD/
├── fishBEVDet/              # BEVDet adapted with MEI distortion-aware VTM
├── fishbevformer/           # BEVFormer adapted with MEI distortion-aware VTM
├── fishPETR/                # PETR adapted with MEI distortion-aware 3D PE
├── kitti360_nusc_converter/ # KITTI-360 → nuScenes format conversion pipeline
└── nuscenes-devkit/         # Customized nuScenes devkit (local, for visualization and evaluation)
```

## Installation
The following instructions set up the environment is only tested for PETR and BEVFormer. For BEVDet, please refer to original environment instructions in the [fishBEVDet](fishBEVDet/README.md), since it uses a different version of mmdetection3d.
### Prerequisites

- Linux (tested on Ubuntu 20.04)
- Python 3.8
- CUDA 11.1
- NVIDIA GPU (experiments run on NVIDIA A5000)

### Step 1: Create and activate conda environment

```bash
conda create -n fishbevod python=3.8 -y
conda activate fishbevod
```

### Step 2: Install PyTorch

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Install mmcv-full

```bash
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### Step 4: Install mmdet and mmseg

```bash
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

### Step 5: Install MMDetection3D

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
cd ..
```

### Step 6: Install additional dependencies

```bash
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13 \
    typing-extensions==4.5.0 pylint ipython==8.12 numpy==1.19.5 \
    matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 \
    scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Step 7: Install the local nuscenes-devkit

This repository ships a customized nuScenes devkit with support for fisheye visualization and mixed-camera configurations. Install it locally instead of the PyPI version:

```bash
cd nuscenes-devkit/setup
pip install -r requirements.txt
cd ..
pip install -e .
cd ..
```

## Dataset Preparation

### 1. Download KITTI-360

Download the KITTI-360 dataset from the [official website](https://www.cvlibs.net/datasets/kitti-360/) and place it under `data/kitti360/`. Since the dataset is coverted to nuScenes format, you can also just name it as 'nuscenes' and `vkitti360-trainval` as `v1.0-trainval` if you prefer.

Expected layout:
```
data/kitti360/
├── calibration/
├── data_2d_raw/
├── data_3d_bboxes/
└── data_poses/
```

### 2. Convert KITTI-360 to nuScenes format

```bash
cd kitti360_nusc_converter

# Full trainval conversion
python convert_kitti360_to_nuscenes.py \
    --kitti360_root ../data/kitti360 \
    --output_dir ../data/kitti360

# (Optional) Create a mini split for debugging
python create_mini.py \
    --input_dir ../data/kitti360 \
    --output_dir ../data/vkitti360-mini

cd ..
```

The converter produces a `vkitti360-trainval` split (~300 scenes, ~68K samples) compatible with the nuScenes devkit.

### 3. (Optional) Rectify fisheye images

To generate the rectified 6-camera baseline:

```bash
cd kitti360_nusc_converter
python rectify_fisheye_images.py \
    --kitti360_root ../data/kitti360 \
    --output_dir ../data/kitti360
cd ..
```

### 4. Create BEVDet data pickle

```bash
cd fishBEVDet
python tools/create_data_bevdet.py \
    --root-path ../data/kitti360 \
    --out-dir ../data/kitti360
cd ..
```

## Training

All models are trained for 24 epochs with a batch size of 8 using AdamW with cosine annealing. Distributed training across multiple NVIDIA A5000 GPUs is recommended.

### FishBEVDet

```bash
cd fishBEVDet
bash tools/dist_train.sh configs/bevdet/bevdet-r50-xxx.py 2 --validate
cd ..
```

### FishBEVFormer (only for bevformer static version without temporal modeling)

```bash
cd fishbevformer
bash tools/dist_train.sh projects/configs/bevformer/bevformer_small_s_xxx.py 2
cd ..
```

### FishPETR

```bash
cd fishPETR
bash tools/dist_train.sh projects/configs/petr/petr_vovnet_gridmask_p4_xxx.py 2
cd ..
```

## Evaluation

```bash
# BEVDet
cd fishBEVDet
bash tools/dist_test.sh <config> <checkpoint> 2 --eval bbox
cd ..

# BEVFormer
cd fishbevformer
bash tools/dist_test.sh <config> <checkpoint> 2 --eval bbox
cd ..

# PETR
cd fishPETR
bash tools/dist_test.sh <config> <checkpoint> 2 --eval bbox
cd ..
```

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026fishbevod,
  title     = {Benchmarking Multi-View BEV Object Detection with Mixed Pinhole and Fisheye Cameras},
  author    = {Liu, Xiangzhong and Shen, Hao},
  journal   = {arXiv preprint arXiv:2603.27818},
  year      = {2026}
}
```

## Acknowledgements

This work builds upon and is grateful to the following projects:

- **[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)** — Liao et al., *KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D*, TPAMI 2022. The dataset and its rich sensor suite (stereo + fisheye cameras with LiDAR annotations) form the foundation of our benchmark.
- **[nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)** — Caesar et al., *nuScenes: A multimodal dataset for autonomous driving*, CVPR 2020. We extend the devkit to support fisheye visualization and mixed-camera evaluation.
- **[PETR](https://github.com/megvii-research/PETR)** — Liu et al., *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*, ECCV 2022. The projection-free BEV architecture that we adapt with MEI distortion-aware 3D positional encoding and polar coordinates.
- **[BEVFormer](https://github.com/fundamentalvision/BEVFormer)** — Li et al., *BEVFormer: Learning Bird's-Eye-View Representation from LiDAR-Camera via Spatiotemporal Transformers*, TPAMI 2024. The backward-projection BEV architecture adapted with MEI-based spatial cross-attention.
- **[BEVDet](https://github.com/HuangJunJie2017/BEVDet)** — Huang et al., *BEVDet: High-performance Multi-Camera 3D Object Detection in Bird-Eye View*, arXiv 2021. The forward-projection BEV architecture adapted with MEI-based depth lifting.

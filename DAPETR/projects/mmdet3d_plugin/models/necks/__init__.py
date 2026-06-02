# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from .cp_fpn import CPFPN
from .cp_fpn_film import CPFPNFilm
__all__ = ['CPFPN', 'CPFPNFilm']

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torch import Tensor

def mipmap_grid_sample_2d(
    x: List[Tensor],
    grid: Tensor,
    vt_dxdy_img: Tensor,
    max_aniso: int,
    padding_mode: int,
    interpolation_mode: int,
    align_corners: bool,
    force_max_ansio: bool,
    clip_grad: bool,
) -> Tensor: ...

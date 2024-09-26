# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torch import Tensor

def grid_scatter_2d(
    input: List[Tensor],
    grid: Tensor,
    output_height: int,
    output_width: int,
    padding_mode: int,
    interpolation_mode: int,
    align_corners: bool,
) -> Tensor: ...

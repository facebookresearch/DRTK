# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor

def msi(
    ray_o: th.Tensor,
    ray_d: th.Tensor,
    texture: th.Tensor,
    sub_step_count: int,
    min_inv_r: float,
    max_inv_r: float,
    stop_thresh: float,
) -> th.Tensor: ...

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

# pyrefly: ignore [missing-module-attribute]
from torch import Tensor

def render_with_texture(
    v: Tensor,
    vi: Tensor,
    uv: Tensor,
    texture: Tensor,
    opacity: Tensor,
    background: Tensor,
    height: int,
    width: int,
    interpolation_mode: int,
    padding_mode: int,
    transmittance_epsilon: float,
    rgb_activation: int,
    alpha_activation: int,
) -> List[Tensor]: ...
def render_with_mlp(
    v: Tensor,
    vi: Tensor,
    uv0: Tensor,
    uv1: Tensor,
    texture0: Tensor,
    texture1: Tensor,
    weights: List[Tensor],
    biases: List[Tensor],
    opacity: Tensor,
    background: Tensor,
    height: int,
    width: int,
    negative_slope: float,
    interpolation_mode: int,
    padding_mode: int,
    transmittance_epsilon: float,
    rgb_activation: int,
    alpha_activation: int,
) -> List[Tensor]: ...

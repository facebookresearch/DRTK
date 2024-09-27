# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from drtk.utils import load_torch_ops

load_torch_ops("drtk.msi_ext")


@th.compiler.disable
def msi(
    ray_o: th.Tensor,
    ray_d: th.Tensor,
    texture: th.Tensor,
    sub_step_count: int = 2,
    min_inv_r: float = 1.0,
    max_inv_r: float = 0.0,
    stop_thresh: float = 1e-7,
) -> th.Tensor:
    """
    Renders a Multi-Sphere Image which is similar to the one described in "NeRF++: Analyzing and Improving
    Neural Radiance Fields"
    The implementation performs bilinear sampling in the spatial dimensions of each layer and cubic between
    the layers.

    Args:
        ray_o (th.Tensor): Ray origins [N x 3]

        ray_d (th.Tensor): Ray directions [N x 3]

        texture (th.Tensor): The MSI texture [L x 4 x H x W], where L - number of layers.
            The first 3 channels are the color channels, and the fourth one is the sigma (transmittance)
            channel (negative log of alpha).

        sub_step_count (int, optional): Rate of the subsampling of the layers. Default is 2.

        min_inv_r (float, optional): Inverse of the minimum sphere radius. Default is 1 for unit radius.

        max_inv_r (float, optional): Inverse of the maximum sphere radius. Default is 0 for infinite radius.

        stop_thresh (bool, optional): The threshold for early ray termination when the accumulated
            transmittance goes beyond the specified value.

    Returns:
        output (Tensor): Result of the sampled MSI. First three channels are the color channels, and the 4th
        one is sigma (transmittance). [N x 4]
    """
    return th.ops.msi_ext.msi(
        ray_o, ray_d, texture, sub_step_count, min_inv_r, max_inv_r, stop_thresh
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor

def interpolate(
    vert_attributes: Tensor,
    vi: Tensor,
    index_img: Tensor,
    bary_img: Tensor,
) -> Tensor: ...

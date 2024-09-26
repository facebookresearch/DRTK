# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor

def edge_grad_estimator(
    v_pix: Tensor,
    v_pix_img: Tensor,
    vi: Tensor,
    img: Tensor,
    index_img: Tensor,
) -> Tensor: ...

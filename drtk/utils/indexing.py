# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch as th


def index(x: th.Tensor, idxs: th.Tensor, dim: int) -> th.Tensor:
    """Index a tensor along a given dimension using an index tensor, replacing
    the shape along the given dimension with the shape of the index tensor.

    Example:
    x:    [8, 7306, 3]
    idxs: [11000, 3]

    y = index(x, idxs, dim=1) -> y: [8, 11000, 3, 3]
    with each y[b, i, j, k] = x[b, idxs[i, j], k]

    """

    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*idxs.shape]
    return x.index_select(dim, idxs.view(-1)).reshape(target_shape)

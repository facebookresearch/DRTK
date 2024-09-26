# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torch import Tensor

def rasterize(
    v: Tensor,
    vi: Tensor,
    height: int,
    width: int,
) -> List[Tensor]: ...

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torch import Tensor

def render(
    v: Tensor,
    vi: Tensor,
    index_img: Tensor,
) -> List[Tensor]: ...

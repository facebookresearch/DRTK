# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torch import Tensor

def render(
    v: Tensor,
    vi: Tensor,
    index_img: Tensor,
) -> List[Tensor]: ...

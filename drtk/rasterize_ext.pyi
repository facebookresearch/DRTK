from typing import List

from torch import Tensor

def rasterize(
    v: Tensor,
    vi: Tensor,
    height: int,
    width: int,
) -> List[Tensor]: ...

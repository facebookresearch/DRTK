from typing import List

from torch import Tensor

def render(
    v: Tensor,
    vi: Tensor,
    index_img: Tensor,
) -> List[Tensor]: ...

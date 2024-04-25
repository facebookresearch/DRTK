from torch import Tensor

def interpolate(
    vert_attributes: Tensor,
    vi: Tensor,
    index_img: Tensor,
    bary_img: Tensor,
) -> Tensor: ...

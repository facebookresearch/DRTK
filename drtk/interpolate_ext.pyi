from torch import Tensor

def compute_vert_image(
    vert_attributes: th.Tensor,
    vi: th.Tensor,
    index_img: th.Tensor,
    bary_img: th.Tensor,
) -> th.Tensor: ...

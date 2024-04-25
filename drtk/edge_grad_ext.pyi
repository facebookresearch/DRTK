from torch import Tensor

def edge_grad_estimator(
    v_pix: Tensor,
    v_pix_img: Tensor,
    vi: Tensor,
    img: Tensor,
    index_img: Tensor,
) -> Tensor: ...

from typing import List, Optional

from torch import Tensor

def render_forward(
    v2d: Tensor,
    vt: Tensor,
    vn: Optional[Tensor],
    vi: Tensor,
    vti: Tensor,
    indeximg: Tensor,
    depthimg: Tensor,
    baryimg: Tensor,
    uvimg: Tensor,
    vnimg: Optional[Tensor],
) -> List[Tensor]: ...
def render_backward(
    v2d: Tensor,
    vt: Tensor,
    vn: Optional[Tensor],
    vi: Tensor,
    vti: Tensor,
    indeximg: Tensor,
    grad_depthimg: Tensor,
    grad_baryimg: Tensor,
    grad_uvimg: Tensor,
    grad_vnimg: Optional[Tensor],
    grad_v2d: Tensor,
    grad_vn: Optional[Tensor],
) -> List[Tensor]: ...

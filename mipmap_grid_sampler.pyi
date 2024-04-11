from typing import List

from torch import Tensor

def mipmap_grid_sampler_2d(
    x: List[Tensor],
    grid: Tensor,
    vt_dxdy_img: Tensor,
    max_aniso: int,
    padding_mode: int,
    interpolation_mode: int,
    align_corners: bool,
    force_max_ansio: bool,
    clip_grad: bool,
) -> Tensor: ...

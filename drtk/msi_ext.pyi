from torch import Tensor

def msi_bkg(
    ray_o: th.Tensor,
    ray_d: th.Tensor,
    texture: th.Tensor,
    sub_step_count: int,
    min_inv_r: float,
    max_inv_r: float,
    stop_thresh: float,
) -> th.Tensor: ...

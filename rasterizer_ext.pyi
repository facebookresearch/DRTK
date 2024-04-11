from typing import Any, Dict

from torch import Tensor

def rasterize_packed(
    v: Tensor,
    vi: Tensor,
    depth_img: Tensor,
    index_img: Tensor,
    packed_index_img: Tensor,
) -> int: ...

class VulkanRasterizerContext:
    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self, height: int, width: int, device_idx: int, backface_culling: bool
    ): ...
    # pyre-fixme[3]: Return type must be annotated.
    def to(self, device_idx: int): ...
    # pyre-fixme[3]: Return type must be annotated.
    def rasterize(
        self, v: Tensor, index_img: Tensor, depth_img: Tensor, write_depth: bool
    ): ...
    # pyre-fixme[3]: Return type must be annotated.
    def updateTopology(self, vi: Tensor): ...
    def __getstate__(self) -> Dict[str, Any]: ...
    # pyre-fixme[3]: Return type must be annotated.
    def __setstate__(self, state: Dict[str, Any]): ...

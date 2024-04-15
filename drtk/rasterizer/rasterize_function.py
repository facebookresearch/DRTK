# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch as th

from .. import rasterizer_ext

th.ops.load_library(rasterizer_ext.__file__)

from typing import List, Tuple

from torch import Tensor


class Rasterize(th.autograd.Function):
    @staticmethod
    @th.cuda.amp.custom_fwd(cast_inputs=th.float32)
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(ctx, vkcm, v_clip, depth_img, index_img, write_depth):
        vkcm.rasterize(v_clip, index_img, depth_img, write_depth)

        ctx.mark_dirty(depth_img, index_img)
        ctx.mark_non_differentiable(depth_img, index_img)
        return depth_img, index_img

    @staticmethod
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, *args) -> List[None]:
        return [None for _ in args]


class PackedRasterize(th.autograd.Function):
    @staticmethod
    @th.cuda.amp.custom_fwd(cast_inputs=th.float32)
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(ctx, v_pix, vi, depth_img, index_img, packed_index_img):
        th.ops.rasterizer_ext.rasterize_packed(
            v_pix, vi, depth_img, index_img, packed_index_img
        )

        ctx.mark_dirty(depth_img, index_img, packed_index_img)
        ctx.mark_non_differentiable(depth_img, index_img, packed_index_img)
        return depth_img, index_img, packed_index_img

    @staticmethod
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, *args) -> List[None]:
        return [None for _ in args]


# pyre-fixme[16] Undefined attribute: `Rasterize` has no attribute `apply`.
# pyre-fixme[5]: Global expression must be annotated.
rasterize = Rasterize.apply
# pyre-fixme[16] Undefined attribute: `PackedRasterize` has no attribute `apply`.
# pyre-fixme[5]: Global expression must be annotated.
rasterize_packed = PackedRasterize.apply

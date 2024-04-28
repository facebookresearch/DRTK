# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Optional, Tuple

import torch as th

from drtk import render_ext
from torch import Tensor

th.ops.load_library(render_ext.__file__)
# pyre-fixme[5]: Global expression must be annotated.
render_forward = th.ops.render_ext.render_forward
# pyre-fixme[5]: Global expression must be annotated.
render_backward = th.ops.render_ext.render_backward


class CudaRenderer(th.autograd.Function):
    @staticmethod
    @th.cuda.amp.custom_fwd(cast_inputs=th.float32)
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(ctx, v2d, vt, vi, vti, index_img, vn=None):
        N, H, W = index_img.shape
        depth_img = th.empty((N, H, W), device=v2d.device, dtype=v2d.dtype)
        bary_img = th.empty((N, H, W, 3), device=v2d.device, dtype=v2d.dtype)
        uv_img = th.empty((N, H, W, 2), device=v2d.device, dtype=v2d.dtype)

        vn_img = None
        if vn is not None:
            vn_img = th.empty((N, H, W, 3), device=v2d.device)

        assert v2d.dtype == th.float
        assert vt.dtype == th.float
        assert vi.dtype == th.int
        assert vti.dtype == th.int
        assert index_img.dtype == th.int
        render_forward(
            v2d, vt, vn, vi, vti, index_img, depth_img, bary_img, uv_img, vn_img
        )
        ctx.save_for_backward(v2d, vt, vn, vi, vti, index_img)

        outs = [depth_img, bary_img, uv_img]
        if vn is not None:
            outs.append(vn_img)
        return tuple(outs)

    @staticmethod
    @th.cuda.amp.custom_bwd
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        # pyre-fixme[2]: Parameter must be annotated.
        grad_depth_img,
        # pyre-fixme[2]: Parameter must be annotated.
        grad_bary_img,
        # pyre-fixme[2]: Parameter must be annotated.
        grad_uv_img,
        # pyre-fixme[2]: Parameter must be annotated.
        grad_vn_img=None,
    ) -> Tuple[Tensor, None, None, None, None, Optional[Tensor]]:
        v2d, vt, vn, vi, vti, index_img = ctx.saved_tensors
        grad_v2d = th.zeros_like(v2d)

        grad_vn = None
        if vn is not None:
            assert grad_vn_img is not None
            grad_vn = th.zeros_like(vn)
            grad_vn_img = grad_vn_img.contiguous()

        render_backward(
            v2d,
            vt,
            vn,
            vi,
            vti,
            index_img,
            grad_depth_img.contiguous(),
            grad_bary_img.contiguous(),
            grad_uv_img.contiguous(),
            grad_vn_img,
            grad_v2d,
            grad_vn,
        )

        return grad_v2d, None, None, None, None, grad_vn


# pyre-fixme[16] Undefined attribute: `CudaRenderer` has no attribute `apply`.
# pyre-fixme[5]: Global expression must be annotated.
render = CudaRenderer.apply

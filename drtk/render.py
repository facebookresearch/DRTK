# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from typing import Tuple

import torch as th
from drtk.utils import load_torch_ops

load_torch_ops("drtk.render_ext")


@th.compiler.disable
def render(
    v: th.Tensor,
    vi: th.Tensor,
    index_img: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor]:
    depth_img, bary_img = th.ops.render_ext.render(v, vi, index_img)
    return depth_img, bary_img


def index(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)


@lru_cache
def _get_grid(width: int, height: int, device: th.device):
    return th.stack(
        th.meshgrid(th.arange(height, device=device), th.arange(width, device=device))[
            ::-1
        ],
        dim=2,
    )


def render_ref(
    v: th.Tensor, vi: th.Tensor, index_img: th.Tensor
) -> Tuple[th.Tensor, th.Tensor]:
    # Run reference implementation in double precision to get as good reference as possible

    orig_dtype = v.dtype
    v = v.double()
    b = v.shape[0]
    mask = th.ne(index_img, -1)
    float_mask = mask.float()[:, None]
    index_img_clamped = index_img.clamp(min=0).long()

    grid = _get_grid(index_img.shape[-1], index_img.shape[-2], device=v.device)

    # compute barycentric coordinates
    vi_img = index(vi, index_img_clamped, 0).long()
    v_img0 = th.cat(
        [index(v[i], vi_img[i, ..., 0].data, 0)[None, ...] for i in range(b)], dim=0
    )
    v_img1 = th.cat(
        [index(v[i], vi_img[i, ..., 1].data, 0)[None, ...] for i in range(b)], dim=0
    )
    v_img2 = th.cat(
        [index(v[i], vi_img[i, ..., 2].data, 0)[None, ...] for i in range(b)], dim=0
    )

    vec01 = v_img1 - v_img0
    vec02 = v_img2 - v_img0
    vec12 = v_img2 - v_img1

    def epsclamp(x: th.Tensor) -> th.Tensor:
        return th.where(x < 0, x.clamp(max=-1e-16), x.clamp(min=1e-16))

    det = vec01[..., 0] * vec02[..., 1] - vec01[..., 1] * vec02[..., 0]
    denominator = epsclamp(det)

    vp0 = grid[None, ...] - v_img0[..., :2]
    vp1 = grid[None, ...] - v_img1[..., :2]

    lambda_0 = (vp1[..., 1] * vec12[..., 0] - vp1[..., 0] * vec12[..., 1]) / denominator
    lambda_1 = (vp0[..., 0] * vec02[..., 1] - vp0[..., 1] * vec02[..., 0]) / denominator
    lambda_2 = (vp0[..., 1] * vec01[..., 0] - vp0[..., 0] * vec01[..., 1]) / denominator

    assert th.allclose(lambda_0 + lambda_1 + lambda_2, th.ones_like(lambda_0))

    lambda_0_mul_w0 = lambda_0 / epsclamp(v_img0[:, :, :, 2])
    lambda_1_mul_w1 = lambda_1 / epsclamp(v_img1[:, :, :, 2])
    lambda_2_mul_w2 = lambda_2 / epsclamp(v_img2[:, :, :, 2])
    zi = 1.0 / epsclamp(lambda_0_mul_w0 + lambda_1_mul_w1 + lambda_2_mul_w2)

    bary_0 = lambda_0_mul_w0 * zi
    bary_1 = lambda_1_mul_w1 * zi
    bary_2 = lambda_2_mul_w2 * zi

    bary_img = (
        th.cat(
            (bary_0[:, None, :, :], bary_1[:, None, :, :], bary_2[:, None, :, :]),
            dim=1,
        )
        * float_mask
    )

    depth_img = zi * float_mask[:, 0]

    return depth_img.to(orig_dtype), bary_img.to(orig_dtype)

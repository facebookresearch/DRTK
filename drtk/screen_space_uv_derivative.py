# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

import torch as th

from drtk.interpolate import interpolate

from drtk.utils import face_dpdt, project_points_grad


def screen_space_uv_derivative(
    v: th.Tensor,
    vt: th.Tensor,
    vi: th.Tensor,
    vti: th.Tensor,
    index_img: th.Tensor,
    bary_img: th.Tensor,
    mask: th.Tensor,
    campos: th.Tensor,
    camrot: th.Tensor,
    focal: th.Tensor,
    dist_mode: Optional[Sequence[str]] = None,
    dist_coeff: Optional[th.Tensor] = None,
) -> th.Tensor:
    """
    Computes per-pixel uv derivative - vt_dxdy_img with respect to the pixel-space position.
    vt_dxdy_img is an image of 2x2 Jacobian matrices of the form: [[du/dx, dv/dx],
                                                                   [du/dy, dv/dy]]
    Shape: N x H x W x 2 x 2
    """

    dpdt_t, vf = face_dpdt(v, vt, vi.long(), vti.long())

    # make three of this for each vertex of the face
    dpdt_t = dpdt_t[:, :, None]
    dpdt_t = dpdt_t.expand(-1, -1, 3, -1, -1)

    # UV grads are not quite well interpolated with barycentrics
    # We computes uv grads at per-pixel basis. For faster compute it is better to use CUDA kernel

    # make new index list, because gradients have discontinuities and we do not want to interpolate them
    vi_dis = th.arange(0, 3 * vi.shape[0], dtype=th.int32, device=v.device).view(-1, 3)
    dpdt_t_img = interpolate(
        dpdt_t.reshape(dpdt_t.shape[0], dpdt_t.shape[1] * dpdt_t.shape[2], -1),
        vi_dis,
        index_img,
        bary_img,
    ).permute(0, 2, 3, 1)
    dpdt_t_img = dpdt_t_img.view(*dpdt_t_img.shape[:3], 2, 3)
    vf_img = interpolate(
        vf.reshape(vf.shape[0], vf.shape[1] * vf.shape[2], -1),
        vi_dis,
        index_img,
        bary_img,
    ).permute(0, 2, 3, 1)
    # duplicate vertex position vector for u and v
    vf_img = vf_img[:, :, :, None].expand(-1, -1, -1, 2, -1)
    # Compute 2D pixel-space gradients (d p_pix / dt)^T.
    dp_pix_dt_t_img = project_points_grad(
        dpdt_t_img.reshape(v.shape[0], -1, 3),
        vf_img.reshape(v.shape[0], -1, 3),
        campos,
        camrot,
        focal,
        dist_mode,
        dist_coeff,
    )
    # Uncollapse dimension. The result is (d p_pix / dt)^T
    # Where: dp_pix_dt[..., i, j] = d p_pix[j] / dt[i]
    dp_pix_dt_t_img = dp_pix_dt_t_img.view(*dpdt_t_img.shape[:3], 2, 2)
    # Inverse Jacobian: (dt / d p_pix)^T = ((d p_pix / dt)^T)^-1
    # Where: dt_dp_pix_t[..., i, j] = dt[j] / dp_pix[i]
    vt_dxdy_img, _ = th.linalg.inv_ex(dp_pix_dt_t_img)
    # pyre-fixme[16] Undefined attribute: `th.Tensor` has no attribute `__invert__`.
    vt_dxdy_img[~mask, :, :] = 0
    return vt_dxdy_img

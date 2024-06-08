# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple

import torch as th
import torch.nn.functional as thf
from drtk import edge_grad_ext
from drtk.interpolate import interpolate
from drtk.utils import index


th.ops.load_library(edge_grad_ext.__file__)


def edge_grad_estimator(
    v_pix: th.Tensor,
    vi: th.Tensor,
    bary_img: th.Tensor,
    img: th.Tensor,
    index_img: th.Tensor,
    v_pix_img_hook: Optional[Callable[[th.Tensor], None]] = None,
) -> th.Tensor:
    """
    Args:
        v_pix: Pixel-space vertex coordinates with preserved camera-space Z-values.
            N x V x 3

        vi: face vertex index list tensor
            V x 3

        bary_img: 3D barycentric coordinate image tensor
            N x 3 x H x W

        img: The rendered image
            N x C x H x W

        index_img: index image tensor
            N x H x W

        v_pix_img_hook: a backward hook that will be registered to v_pix_img. Useful for examining
            generated image space. Default None

    Returns:
        returns the img argument unchanged. Optionally also returns computed
        v_pix_img. Your loss should use the returned img, even though it is
        unchanged.

    Note:
        It is important to not spatially modify the rasterized image before passing it to edge_grad_estimator.
        Any operation as long as it is differentiable is ok after the edge_grad_estimator.

        Examples of opeartions that can be done before edge_grad_estimator:
            - Pixel-wise MLP
            - Color mapping
            - Color correction, gamma correction
        If the operation is significantly non-linear, then it is recommended to do it before
        edge_grad_estimator. All sorts of clipping and clamping (e.g. `x.clamp(min=0.0, max=1.0)`), must be
        done before edge_grad_estimator.

        Examples of operations that are not allowed before edge_grad_estimator:
            - Gaussian blur
            - Warping, deformation
            - Masking, cropping, making holes.

    Usage::

        from drtk.renderlayer import edge_grad_estimator

        ...

        out = renderlayer(v, tex, campos, camrot, focal, princpt,
                 output_filters=["index_img", "render", "mask", "bary_img", "v_pix"])

        img = out["render"] * out["mask"]

        img = edge_grad_estimator(
            v_pix=out["v_pix"],
            vi=rl.vi,
            bary_img=out["bary_img"],
            img=img,
            index_img=out["index_img"]
        )

        optim.zero_grad()
        image_loss = loss_func(img, img_gt)
        image_loss.backward()
        optim.step()
    """

    # Could use v_pix_img output from DRTK, but bary_img needs to be detached.
    v_pix_img = interpolate(v_pix, vi, index_img, bary_img.detach())

    img = th.ops.edge_grad_ext.edge_grad_estimator(v_pix, v_pix_img, vi, img, index_img)

    if v_pix_img_hook is not None:
        v_pix_img.register_hook(v_pix_img_hook)
    return img


def edge_grad_estimator_ref(
    v_pix: th.Tensor,
    vi: th.Tensor,
    bary_img: th.Tensor,
    img: th.Tensor,
    index_img: th.Tensor,
    v_pix_img_hook: Optional[Callable[[th.Tensor], None]] = None,
) -> th.Tensor:
    """
    Python reference implementation for
    :func:`drtk.edge_grad_estimator.edge_grad_estimator`.
    """

    # could use v_pix_img output from DRTK, but bary_img needs to be detached.
    v_pix_img = interpolate(v_pix, vi, index_img, bary_img.detach())
    # pyre-fixme[16]: `EdgeGradEstimatorFunction` has no attribute `apply`.
    img = EdgeGradEstimatorFunction.apply(v_pix, v_pix_img, vi, img, index_img)

    if v_pix_img_hook is not None:
        v_pix_img.register_hook(v_pix_img_hook)
    return img


class EdgeGradEstimatorFunction(th.autograd.Function):
    @staticmethod
    @th.cuda.amp.custom_fwd(cast_inputs=th.float32)
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        v_pix: th.Tensor,
        v_pix_img: th.Tensor,
        vi: th.Tensor,
        img: th.Tensor,
        index_img: th.Tensor,
    ) -> th.Tensor:
        ctx.save_for_backward(v_pix, img, index_img, vi)
        return img

    @staticmethod
    @th.cuda.amp.custom_bwd
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(ctx, grad_output: th.Tensor) -> Tuple[
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
        Optional[th.Tensor],
    ]:
        # early exit in case geometry is not optimized.
        if not ctx.needs_input_grad[1]:
            return None, None, None, grad_output, None

        v_pix, img, index_img, vi = ctx.saved_tensors

        x_grad = img[:, :, :, 1:] - img[:, :, :, :-1]
        y_grad = img[:, :, 1:, :] - img[:, :, :-1, :]

        l_index = index_img[:, None, :, :-1]
        r_index = index_img[:, None, :, 1:]
        t_index = index_img[:, None, :-1, :]
        b_index = index_img[:, None, 1:, :]

        x_mask = r_index != l_index
        y_mask = b_index != t_index

        x_both_triangles = (r_index != -1) & (l_index != -1)
        y_both_triangles = (b_index != -1) & (t_index != -1)

        iimg_clamped = index_img.clamp(min=0).long()

        # compute barycentric coordinates
        b = v_pix.shape[0]
        vi_img = index(vi, iimg_clamped, 0).long()
        p0 = th.cat(
            [index(v_pix[i], vi_img[i, ..., 0].data, 0)[None, ...] for i in range(b)],
            dim=0,
        )
        p1 = th.cat(
            [index(v_pix[i], vi_img[i, ..., 1].data, 0)[None, ...] for i in range(b)],
            dim=0,
        )
        p2 = th.cat(
            [index(v_pix[i], vi_img[i, ..., 2].data, 0)[None, ...] for i in range(b)],
            dim=0,
        )

        v10 = p1 - p0
        v02 = p0 - p2
        n = th.cross(v02, v10)

        px, py = th.meshgrid(
            th.arange(img.shape[-2], device=v_pix.device),
            th.arange(img.shape[-1], device=v_pix.device),
        )

        def epsclamp(x: th.Tensor) -> th.Tensor:
            return th.where(x < 0, x.clamp(max=-1e-8), x.clamp(min=1e-8))

        # pyre-fixme[53]: Captured variable `n` is not annotated.
        # pyre-fixme[53]: Captured variable `p0` is not annotated.
        # pyre-fixme[53]: Captured variable `px` is not annotated.
        # pyre-fixme[53]: Captured variable `py` is not annotated.
        # pyre-fixme[53]: Captured variable `v02` is not annotated.
        # pyre-fixme[53]: Captured variable `v10` is not annotated.
        def check_if_point_inside_triangle(offset_x: int, offset_y: int) -> th.Tensor:
            _px = px + offset_x
            _py = py + offset_y

            vp0p = th.stack([p0[..., 0] - _px, p0[..., 1] - _py], dim=-1) / epsclamp(
                n[..., 2:3]
            )

            bary_1 = v02[..., 0] * -vp0p[..., 1] + v02[..., 1] * vp0p[..., 0]
            bary_2 = v10[..., 0] * -vp0p[..., 1] + v10[..., 1] * vp0p[..., 0]

            return ((bary_1 > 0) & (bary_2 > 0) & ((bary_1 + bary_2) < 1))[:, None]

        left_pnt_inside_right_triangle = (
            check_if_point_inside_triangle(-1, 0)[..., :, 1:]
            & x_mask
            & x_both_triangles
        )
        right_pnt_inside_left_triangle = (
            check_if_point_inside_triangle(1, 0)[..., :, :-1]
            & x_mask
            & x_both_triangles
        )
        down_pnt_inside_up_triangle = (
            check_if_point_inside_triangle(0, 1)[..., :-1, :]
            & y_mask
            & y_both_triangles
        )
        up_pnt_inside_down_triangle = (
            check_if_point_inside_triangle(0, -1)[..., 1:, :]
            & y_mask
            & y_both_triangles
        )

        horizontal_intersection = (
            right_pnt_inside_left_triangle & left_pnt_inside_right_triangle
        )
        vertical_intersection = (
            down_pnt_inside_up_triangle & up_pnt_inside_down_triangle
        )

        left_hangs_over_right = left_pnt_inside_right_triangle & (
            ~right_pnt_inside_left_triangle
        )
        right_hangs_over_left = right_pnt_inside_left_triangle & (
            ~left_pnt_inside_right_triangle
        )

        up_hangs_over_down = up_pnt_inside_down_triangle & (
            ~down_pnt_inside_up_triangle
        )
        down_hangs_over_up = down_pnt_inside_up_triangle & (
            ~up_pnt_inside_down_triangle
        )

        x_grad *= x_mask
        y_grad *= y_mask

        grad_output_x = 0.5 * (grad_output[:, :, :, 1:] + grad_output[:, :, :, :-1])
        grad_output_y = 0.5 * (grad_output[:, :, 1:, :] + grad_output[:, :, :-1, :])

        x_grad = (x_grad * grad_output_x).sum(dim=1)
        y_grad = (y_grad * grad_output_y).sum(dim=1)

        x_grad_no_int = x_grad * (~horizontal_intersection[:, 0])
        y_grad_no_int = y_grad * (~vertical_intersection[:, 0])

        x_grad_spread = th.zeros(
            *x_grad_no_int.shape[:1],
            x_grad_no_int.shape[1],
            y_grad_no_int.shape[2],
            dtype=x_grad_no_int.dtype,
            device=x_grad_no_int.device,
        )
        x_grad_spread[:, :, :-1] = x_grad_no_int * (~right_hangs_over_left[:, 0])
        x_grad_spread[:, :, 1:] += x_grad_no_int * (~left_hangs_over_right[:, 0])

        y_grad_spread = th.zeros_like(x_grad_spread)
        y_grad_spread[:, :-1, :] = y_grad_no_int * (~down_hangs_over_up[:, 0])
        y_grad_spread[:, 1:, :] += y_grad_no_int * (~up_hangs_over_down[:, 0])

        # Intersections. Compute border sliding gradients
        #################################################
        z_grad_spread = th.zeros_like(x_grad_spread)
        x_grad_int = x_grad * horizontal_intersection[:, 0]
        y_grad_int = y_grad * vertical_intersection[:, 0]

        n = thf.normalize(n, dim=-1)
        n = n.permute(0, 3, 1, 2)

        n_left = n[..., :, :-1]
        n_right = n[..., :, 1:]
        n_up = n[..., :-1, :]
        n_down = n[..., 1:, :]

        def get_dp_db(v_varying: th.Tensor, v_fixed: th.Tensor) -> th.Tensor:
            """
            Computes derivative of the point position with respect to edge displacement
            See drtk/src/edge_grad/edge_grad_kernel.cu
            Please refer to the paper "Rasterized Edge Gradients: Handling Discontinuities Differentiably"
            for details.
            """

            v_varying = thf.normalize(v_varying, dim=1)
            v_fixed = thf.normalize(v_fixed, dim=1)
            b = th.stack([-v_fixed[:, 1], v_fixed[:, 0]], dim=1)
            b_dot_varying = (b * v_varying).sum(dim=1, keepdim=True)
            return b[:, 0:1] / epsclamp(b_dot_varying) * v_varying

        # We compute partial derivatives by fixing one triangle and moving the
        # other, and then vice versa.

        # Left triangle moves, right fixed
        dp_dbx = get_dp_db(n_left[:, [0, 2]], -n_right[:, [0, 2]])
        x_grad_spread[:, :, :-1] += x_grad_int * dp_dbx[:, 0]
        z_grad_spread[:, :, :-1] += x_grad_int * dp_dbx[:, 1]

        # Left triangle fixed, right moves
        dp_dbx = get_dp_db(n_right[:, [0, 2]], n_left[:, [0, 2]])
        x_grad_spread[:, :, 1:] += x_grad_int * dp_dbx[:, 0]
        z_grad_spread[:, :, 1:] += x_grad_int * dp_dbx[:, 1]

        # Upper triangle moves, lower fixed
        dp_dby = get_dp_db(n_up[:, [1, 2]], -n_down[:, [1, 2]])
        y_grad_spread[:, :-1, :] += y_grad_int * dp_dby[:, 0]
        z_grad_spread[:, :-1, :] += y_grad_int * dp_dby[:, 1]

        # Lower triangle moves, upper fixed
        dp_dby = get_dp_db(n_down[:, [1, 2]], n_up[:, [1, 2]])
        y_grad_spread[:, 1:, :] += y_grad_int * dp_dby[:, 0]
        z_grad_spread[:, 1:, :] += y_grad_int * dp_dby[:, 1]

        m = index_img == -1
        x_grad_spread[m] = 0.0
        y_grad_spread[m] = 0.0

        grad_v_pix = -th.stack([x_grad_spread, y_grad_spread, z_grad_spread], dim=3)

        return None, grad_v_pix, None, grad_output, None

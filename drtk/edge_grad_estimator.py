# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple

import torch as th
import torch.nn.functional as thf
from drtk.interpolate import interpolate
from drtk.utils import index, load_torch_ops


load_torch_ops("drtk.edge_grad_ext")


@th.compiler.disable
def edge_grad_estimator(
    v_pix: th.Tensor,
    vi: th.Tensor,
    bary_img: th.Tensor,
    img: th.Tensor,
    index_img: th.Tensor,
    v_pix_img_hook: Optional[Callable[[th.Tensor], None]] = None,
) -> th.Tensor:
    """Makes the rasterized image ``img`` differentiable at visibility discontinuities
    and backpropagates the gradients to ``v_pix``.

    This function takes a rasterized image ``img`` that is assumed to be differentiable at
    continuous regions but not at discontinuities. In some cases, ``img`` may not be differentiable
    at all. For example, if the image is a rendered segmentation mask, it remains constant at
    continuous regions, making it non-differentiable. However, ``edge_grad_estimator`` can still
    compute gradients at the discontinuities with respect to ``v_pix``.

    The arguments ``bary_img`` and ``index_img`` must correspond exactly to the rasterized image
    ``img``. Each pixel in ``img`` should correspond to a fragment originated prom primitive
    specified by ``index_img`` and it should have barycentric coordinates specified by
    ``bary_img``. This means that with a small change to ``v_pix``, the pixels in ``img`` should
    change accordingly. A frequent mistake that violates this condition is applying a mask
    to the rendered image to exclude unwanted regions, which leads to erroneous gradients.

    The function returns the ``img`` unchanged but with added differentiability at the
    discontinuities. Note that it is not necessary for the input ``img`` to require gradients,
    but the returned ``img`` will always require gradients.

    Args:
        v_pix (Tensor): Pixel-space vertex coordinates, preserving the original camera-space
            Z-values. Shape: :math:`(N, V, 3)`.
        vi (Tensor): Face vertex index list tensor. Shape: :math:`(V, 3)`.
        bary_img (Tensor): 3D barycentric coordinate image tensor. Shape: :math:`(N, 3, H, W)`.
        img (Tensor): The rendered image. Shape: :math:`(N, C, H, W)`.
        index_img (Tensor): Index image tensor. Shape: :math:`(N, H, W)`.
        v_pix_img_hook (Optional[Callable[[th.Tensor], None]]): An optional backward hook that will
            be registered to ``v_pix_img``. Useful for examining the generated image space. Default
            is None.

    Returns:
        Tensor: Returns the input ``img`` unchanged. However, the returned image now has added
        differentiability at visibility discontinuities. This returned image should be used for
        computing losses

    Note:
        It is crucial not to spatially modify the rasterized image before passing it to
        `edge_grad_estimator`. That stems from the requirement that ``bary_img`` and ``index_img``
        must correspond exactly to the rasterized image ``img``. That means that the location of all
        discontinuities is controlled by ``v_pix`` and can be modified by modifing ``v_pix``.

        Operations that are allowed, as long as they are differentiable, include:
            - Pixel-wise MLP
            - Color mapping
            - Color correction, gamma correction
            - Anything that would be indistinguishable from processing fragments independently
              before their values get assigned to pixels of ``img``

        Operations that **must be avoided** before `edge_grad_estimator` include:
            - Gaussian blur
            - Warping or deformation
            - Masking, cropping, or introducing holes

        There is however, no issue with appling them after `edge_grad_estimator`.

        If the operation is highly non-linear, it is recommended to perform it before calling
        :func:`edge_grad_estimator`.
        All sorts of clipping and clamping (e.g., `x.clamp(min=0.0, max=1.0)`) must also be done
        before invoking this function.

    Usage Example::

        import torch.nn.functional as thf
        from drtk import transform, rasterize, render, interpolate, edge_grad_estimator

        ...

        v_pix = transform(v, tex, campos, camrot, focal, princpt)
        index_img = rasterize(v_pix, vi, width=512, height=512)
        _, bary_img = render(v_pix, vi, index_img)
        vt_img = interpolate(vt, vti, index_img, bary_img)

        img = thf.grid_sample(
            tex,
            vt_img.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )

        mask = (index_img != -1)[:, None, :, :]

        img = img * mask

        img = edge_grad_estimator(
            v_pix=v_pix,
            vi=vi,
            bary_img=bary_img,
            img=img,
            index_img=index_img
        )

        optim.zero_grad()
        image_loss = loss_func(img, img_gt)
        image_loss.backward()
        optim.step()
    """

    # TODO: avoid call to interpolate, use backward kernel of interpolate directly
    # Doing so will make `edge_grad_estimator` zero-overhead in forward pass
    # At the moment, value of `v_pix_img` is ignored, and only passed to
    # edge_grad_estimator so that backward kernel can be called with the computed gradient.
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
    :func:`drtk.edge_grad_estimator`.
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

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch as th
import torch.nn.functional as thf
from drtk.renderlayer.geomutils import compute_vert_image

from .geomutils import index


# pyre-fixme[3]: Return type must be annotated.
def edge_grad_estimator(
    # pyre-fixme[2]: Parameter must be annotated.
    v_pix,
    # pyre-fixme[2]: Parameter must be annotated.
    vi,
    # pyre-fixme[2]: Parameter must be annotated.
    bary_img,
    # pyre-fixme[2]: Parameter must be annotated.
    img,
    # pyre-fixme[2]: Parameter must be annotated.
    index_img,
    return_v_pix_img: bool = False,
):
    """
    Python reference implementation for
    :func:`drtk.renderlayer.edge_grad_estimator.edge_grad_estimator`.
    """

    # could use v_pix_img output from DRTK, but bary_img needs to be detached.
    v_pix_img = compute_vert_image(v_pix, vi, index_img, bary_img.detach())
    # pyre-fixme[16]: `EdgeGradEstimatorFunction` has no attribute `apply`.
    img = EdgeGradEstimatorFunction.apply(v_pix, v_pix_img, vi, img, index_img)
    if return_v_pix_img:
        return img, v_pix_img
    return img


class EdgeGradEstimatorFunction(th.autograd.Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(ctx, v_pix, v_pix_img, vi, img, index_img):
        ctx.save_for_backward(v_pix, img, index_img, vi)
        return img

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, grad_output):
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

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def epsclamp(x):
            return th.where(x < 0, x.clamp(max=-1e-8), x.clamp(min=1e-8))

        # pyre-fixme[53]: Captured variable `n` is not annotated.
        # pyre-fixme[53]: Captured variable `p0` is not annotated.
        # pyre-fixme[53]: Captured variable `px` is not annotated.
        # pyre-fixme[53]: Captured variable `py` is not annotated.
        # pyre-fixme[53]: Captured variable `v02` is not annotated.
        # pyre-fixme[53]: Captured variable `v10` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def check_if_point_inside_triangle(offset_x, offset_y):
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

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def get_dp_db(v_moving, v_fixed):
            """
            Computes derivative of the point position with respect to edge displacement

            Args:
                v_moving: Projection of the normal of the movable triangle onto
                    the plane of consideration (XZ or YZ). Shape: N x 3 x H x W

                v_fixed:  Projection of the normal of the fixed triangle onto
                    the plane of consideration (XZ or YZ). Shape: N x 3 x H x W

            Returns: dp/db_x - derivative of the point position (p) with
                respect to edge displacement along X axis (bx)

            This function considers partial derivatives in a XZ or YZ plane.
            For shortness, only XZ plane will be mentioned, but the same
            applies for YZ too.

            We consider movement of edge position along X and along Y
            separately, thus computing partial derivatives.

            Notation:
                p - point position on the movable triangle in XZ plane for
                    which we compute derivatives
                b - edge displacement vector in XZ plane. Displacement of the
                    edge due to displacement of the movable triangle
                v_moving - direction of the displacement vector of the movable
                    triangle, also coincides with its normal
                v_fixed - normal of the fixed triangle
                v_moving' - normalized projection of v_moving onto XZ plane
                v_fixed' - normalized projection of v_fixed onto XZ plane
                t - coordinate on the displacement vector, e.i. the distance
                    that movable triangle was moved along the v_moving

                So: p = p_0 + v_moving' * t

            Since one triangle is fixed, and the other moves, edge displacement
            b will be in plane of the fixed triangle We can compute direction
            of b by rotating v_fixed' by 90 deg.

            Note: We consider only parallel displacement of the movable
            triangle, so p moves along v_moving. We do not need to consider
            rotation, as rotation in XZ plane around the point of intersection
            won't move the point of intersection.

            The derivative is computed as:
                dp/db_x = dp/dt * dt/db_x

            Where:
                dt_db_x = 1 / b_x

            Edge displacement vector b is computed as:
                b_dir = rotate(v_fixed, 90deg)
                b = b_dir / dot(b_dir, v_moving')

            Thus:
                dt/db_x = dot(b_dir, v_moving') / b_dir_x

            The dp/dt part is not normalized projection of v_moving onto XZ plane:
                dp/dt = v_moving_xz
            """

            # Projection of v_moving is not unit length, normalize it
            v_moving_n = thf.normalize(v_moving, dim=1)

            # Projection of v_fixed is not unit length, normalize it
            v_fixed = thf.normalize(v_fixed, dim=1)

            # Rotate normalized v_fixed clock-wise to get direction for b vector. b
            # vector is the displacement of the intersection point cause by
            # moving the movable triangle by v_moving
            b_dir = th.stack([v_fixed[:, 1], -v_fixed[:, 0]], dim=1)

            # Project b_dir onto the normalized v_moving
            b_dir_dot_v_moving = (b_dir * v_moving_n).sum(dim=1, keepdim=True)

            # dt/db_x = dot(b_dir, v_moving') / b_dir_x
            dt_dbx = b_dir_dot_v_moving / epsclamp(b_dir[:, 0:1])

            # dp/db_x = dp/dt * dt/db_x, where dp/dt = v_moving_xz
            dp_dbx = v_moving * dt_dbx
            return dp_dbx

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

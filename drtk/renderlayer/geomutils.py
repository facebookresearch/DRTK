# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch as th
import torch.nn.functional as thf
from drtk.renderlayer.compute_vert_image import compute_vert_image  # noqa
from torch import Tensor


# pyre-fixme[3]: Return type must be annotated.
def face_dpdt(v: th.Tensor, vt: th.Tensor, vi: th.Tensor, vti: th.Tensor):
    """
    Computes transposed Jacobian (dp/dt)^T for each triangle
    Where: p - position of a triangle point
           t - uv coordinates a triangle point

    Args:
        v:  vertex position tensor
            N x V x 3

        vt:  vertex uv tensor
            T x 2

        vi: face vertex position index list tensor
            F x 3

        vti: face vertex uv index list tensor
            F x 3

    Jacobian is computed as:

        dp/dt = dp / db * (dt / db)^-1

        Where b - barycentric coordinates

    However the implementation computes a transposed Jacobian (purely from
    practical perspective - fewer permutations are needed), so the above
    becomes:

        (dp/dt)^T = ((dt / db)^T)^-1 * (dp / db)^T

    Returns:
        dpdt - transposed Jacobian (dp/dt)^T. Shape: N x F x 2 x 3
               Where dpdt[..., i, j] = dp[..., j] / dt[..., i]
        v012 - vertex positions per triangle. Shape: N x F x 3

        where: N - batch size; F - number of triangles
    """

    v012 = v[:, vi]
    vt012 = vt[vti]

    dpdb_t = v012[:, :, 1:3] - v012[:, :, 0:1]
    dtdb_t = vt012[:, 1:3] - vt012[:, 0:1]

    # (db / dt)^T = ((dt / db)^T)^-1
    dbdt_t = th.inverse(dtdb_t)[None, ...]

    # (dp / dt)^T = (db / dt)^T) * (dp / db)^T
    dpdt_t = dbdt_t @ dpdb_t
    return dpdt_t, v012


def face_attribute_to_vert(v: th.Tensor, vi: th.Tensor, attr: th.Tensor) -> Tensor:
    """
    For each vertex, computes a summation of the face attributes to which the
    vertex belongs.
    """
    attr = (
        attr[:, :, None]
        .expand(-1, -1, 3, -1)
        .reshape(attr.shape[0], -1, attr.shape[-1])
    )
    vi_flat = vi.view(vi.shape[0], -1).expand(v.shape[0], -1)
    vattr = th.zeros(v.shape[:-1], dtype=v.dtype, device=v.device)

    vattr = th.stack(
        [vattr.scatter_add(1, vi_flat, attr[..., i]) for i in range(attr.shape[-1])],
        dim=-1,
    )
    return vattr


def vert_binormals(v: Tensor, vt: Tensor, vi: Tensor, vti: Tensor) -> Tensor:
    # Compute  (dp/dt)^T
    dpdt_t, vf = face_dpdt(v, vt, vi, vti)

    # Take the dp/dt.u part. Produces u vector in 3D world-space which we use for binormal vector
    fbnorms = dpdt_t[:, :, 0, :]

    vbnorms = face_attribute_to_vert(v, vi, fbnorms)
    return thf.normalize(vbnorms, dim=-1)

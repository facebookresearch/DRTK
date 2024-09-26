# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, Union

import torch as th
import torch.nn.functional as thf
from drtk.utils.indexing import index
from torch import Tensor

eps = 1e-8


def face_dpdt(
    v: th.Tensor, vt: th.Tensor, vi: th.Tensor, vti: th.Tensor
) -> Tuple[th.Tensor, th.Tensor]:
    """
    This function calculates the transposed Jacobian matrix (∂p/∂t)^T for each triangle.
    Where:
     -  p represents the 3D coordinates of a point on the plane of the triangle,
     -  t denotes the UV coordinates assiciated with the point.

    Args:
        v:  vertex position tensor
            N x V x 3

        vt:  vertex uv tensor
            N x T x 2

        vi: face vertex position index list tensor
            F x 3

        vti: face vertex uv index list tensor
            F x 3

    Jacobian is computed as:

        ∂p/∂t = ∂p / ∂b * (∂t / ∂b)^-1

        Where b - barycentric coordinates

    However the implementation computes a transposed Jacobian (purely from
    practical perspective - fewer permutations are needed), so the above
    becomes:

        (∂p/∂t)^T = ((∂t / ∂b)^T)^-1 * (∂p / ∂b)^T

    Returns:
        dpdt - transposed Jacobian (∂p/∂t)^T. Shape: N x F x 2 x 3
               Where ∂p∂t[..., i, j] = ∂p[..., j] / ∂t[..., i]
        v012 - vertex positions per triangle. Shape: N x F x 3

        where: N - batch size; F - number of triangles
    """

    v012 = v[:, vi]
    vt012 = vt[:, vti]

    dpdb_t = v012[:, :, 1:3] - v012[:, :, 0:1]
    dtdb_t = vt012[:, :, 1:3] - vt012[:, :, 0:1]

    # (db / dt)^T = ((dt / db)^T)^-1
    dbdt_t = th.inverse(dtdb_t)

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


def face_info(
    v: th.Tensor, vi: th.Tensor, to_compute: Optional[List[str]] = None
) -> Union[th.Tensor, Dict[str, th.Tensor]]:
    """Given a set of vertices ``v`` and indices ``vi`` indexing into ``v``
    defining a set of faces, compute face information (normals, edges, face
    areas) for each face.

    Args:
        v: Vertex positions, shape [batch_size, n_vertices, 3]

        vi: Vertex indices, shape [n_faces, 3]

        to_compute: list of desired information. Any of: {normals, edges, areas}, defaults to all.

    Returns:
        Dict: Face information in the following format::

            {
                "normals": shape [batch_size, n_faces, 3]
                "edges":   shape [batch_size, n_faces, 3, 3]
                "areas":   shape [batch_size, n_faces, 1]
            }

        or just one of the above values not in a Dict if only one is
        requested.

    """
    if to_compute is None:
        to_compute = ["normals", "edges", "areas"]

    b = v.shape[0]
    vi = vi.expand(b, -1, -1)

    p0 = th.stack([index(v[i], vi[i, :, 0], 0) for i in range(b)])
    p1 = th.stack([index(v[i], vi[i, :, 1], 0) for i in range(b)])
    p2 = th.stack([index(v[i], vi[i, :, 2], 0) for i in range(b)])
    v0 = p1 - p0
    v1 = p0 - p2

    need_normals = "normals" in to_compute
    need_areas = "areas" in to_compute
    need_edges = "edges" in to_compute

    output = {}

    if need_normals or need_areas:
        normals = th.cross(v1, v0, dim=-1)
        norm = th.linalg.vector_norm(normals, dim=-1, keepdim=True)

        if need_areas:
            output["areas"] = 0.5 * norm

        if need_normals:
            output["normals"] = normals / norm.clamp(min=eps)

    if need_edges:
        v2 = p2 - p1
        output["edges"] = th.stack([v0, v1, v2], dim=2)

    if len(to_compute) == 1:
        return output[to_compute[0]]
    else:
        return output


def vert_binormals(v: Tensor, vt: Tensor, vi: Tensor, vti: Tensor) -> Tensor:
    # Compute  (dp/dt)^T
    dpdt_t, vf = face_dpdt(v, vt, vi, vti)

    # Take the dp/dt.u part. Produces u vector in 3D world-space which we use for binormal vector
    fbnorms = dpdt_t[:, :, 0, :]

    vbnorms = face_attribute_to_vert(v, vi, fbnorms)
    return thf.normalize(vbnorms, dim=-1)


def vert_normals(
    v: th.Tensor, vi: th.Tensor, fnorms: Optional[th.Tensor] = None
) -> th.Tensor:
    """Given a set of vertices ``v`` and indices ``vi`` indexing into ``v``
    defining a set of faces, compute normals for each vertex by averaging the
    face normals for each face which includes that vertex.

    Args:
        v: Vertex positions, shape [batch_size, n_vertices, 3]

        vi: Vertex indices, shape [batch_size, n_faces, 3]

        fnorms: Face normals. Optional, provide them if available, otherwise they will be computed
                from `v` and `vi`. Shape [n_faces, 3]

    Returns:
        th.Tensor: Vertex normals, shape [batch_size, n_vertices, 3]

    """
    if fnorms is None:
        fnorms = face_info(v, vi, ["normals"])
        assert isinstance(fnorms, th.Tensor)
    vnorms = face_attribute_to_vert(v, vi, fnorms)
    return thf.normalize(vnorms, dim=-1)

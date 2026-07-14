# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
``drtk.interpolate`` module provides functions for differentiable interpolation of vertex
attributes across the fragments, e.i. pixels covered by the primitive.
"""

import torch as th
from drtk.utils import load_torch_ops

load_torch_ops("drtk.interpolate_ext")


@th.compiler.disable
def interpolate(
    vert_attributes: th.Tensor,
    vi: th.Tensor,
    index_img: th.Tensor,
    bary_img: th.Tensor,
) -> th.Tensor:
    """
    Performs a linear interpolation of the vertex attributes given the barycentric coordinates

    Args:
        vert_attributes (th.Tensor):  vertex attribute tensor
            N x V x C
        vi (th.Tensor): face vertex index list tensor
            F x 3 or N x F x 3
        index_img (th.Tensor): index image tensor
            N x H x W
        bary_img (th.Tensor): 3D barycentric coordinate image tensor
            N x 3 x H x W

    Returns:
        A tensor with interpolated vertex attributes with a shape [N, C, H, W]

    .. warning::
        The returned tensor has only valid values for pixels which have a valid index in ``index_img``.
        For all other pixels, which had index ``-1`` in ``index_img``, the returned tensor will have non-zero
        values which should be ignored.
    """
    if vi.ndim == 2:
        vi = vi[None].expand(vert_attributes.shape[0], -1, -1)

    return th.ops.interpolate_ext.interpolate(vert_attributes, vi, index_img, bary_img)


@th.compiler.disable
def interpolation_matrix(
    vi: th.Tensor,
    index_img: th.Tensor,
    bary_img: th.Tensor,
    num_vertices: int,
) -> th.Tensor:
    """Build the sparse pixel-to-vertex interpolation matrix.

    This returns the matrix ``A`` that maps per-vertex attributes ``X`` to the
    foreground raster pixels produced by a fixed rasterization:

    .. code-block:: text

        pixel_values = A @ X

    Each valid pixel contributes exactly one row. That row contains three
    non-zero entries: the pixel barycentric weights at the three vertex columns
    of the rasterized triangle. Columns are sorted within each CSR row, so the
    value order may differ from the triangle corner order. Pixels with
    ``index_img == -1`` are omitted entirely, so the row order is the flattened
    ``[N, H, W]`` pixel order with background pixels skipped.

    ``index_img`` is expected to come from DRTK rasterization and contain only
    background ``-1`` or triangle indices in ``[0, F)``. ``num_vertices`` is
    expected to be larger than every vertex index in ``vi``. The native kernels
    do not add extra range-validation work for those contracts; in particular,
    checking ``vi.max()`` on CUDA would add a synchronization only for defensive
    validation before constructing the sparse tensor with
    ``check_invariants=False``. Faces must also have three distinct vertex
    indices; degenerate faces with duplicate corners are unsupported and can
    produce invalid CSR rows with duplicate column entries.

    Differentiability:
        The CSR row/column indices are discrete rasterization/topology data and
        are intentionally non-differentiable. The sparse values are the
        barycentric coordinates, so gradients through operations that support
        sparse CSR values propagate back to ``bary_img``. No gradients are
        produced for ``vi`` or ``index_img``.

    Args:
        vi: Face vertex index tensor, shape ``[F, 3]``, ``[1, F, 3]``,
            or ``[N, F, 3]``. Singleton 3D batches are broadcast to ``N``.
        index_img: Rasterized triangle index image, shape ``[N, H, W]``.
        bary_img: Barycentric image, shape ``[N, 3, H, W]``.
        num_vertices: Number of vertex unknowns, i.e. the number of CSR columns.

    Returns:
        A sparse CSR tensor with shape ``[num_valid_pixels, num_vertices]`` and
        ``3 * num_valid_pixels`` non-zero values.
    """
    if vi.ndim == 2:
        vi = vi[None].expand(index_img.shape[0], -1, -1)
    elif vi.ndim == 3 and vi.shape[0] == 1 and index_img.shape[0] != 1:
        vi = vi.expand(index_img.shape[0], -1, -1)

    crow_indices, col_indices, values, row_pixels = (
        th.ops.interpolate_ext.interpolation_matrix(vi, index_img, bary_img)
    )
    return th.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=(int(row_pixels.numel()), int(num_vertices)),
        device=values.device,
        dtype=values.dtype,
        check_invariants=False,
    )


@th.compiler.disable
def interpolation_normal_matrix(
    vi: th.Tensor,
    index_img: th.Tensor,
    bary_img: th.Tensor,
    num_vertices: int,
) -> th.Tensor:
    """Assemble the sparse normal matrix for barycentric interpolation.

    Let ``A`` be the matrix returned by :func:`interpolation_matrix` for the
    same ``vi``, ``index_img`` and ``bary_img``. This function returns
    ``A.T @ A`` directly, without materializing ``A``. For every foreground
    pixel, the CUDA/CPU kernel accumulates the nine products
    ``bary_i * bary_j`` into the CSR entry corresponding to the owning
    triangle's directed vertex pair ``(vi[i], vi[j])``.

    The sparsity pattern depends only on topology, not on image resolution or
    barycentric values. The C++ op caches that CSR structure and the per-face
    pair lookup per output device for repeated calls with the same face-index
    tensor, so iterative solvers can reuse both the expensive topology analysis
    and the device-resident CSR buffers while still updating numeric values
    every rasterization.

    ``index_img`` is expected to come from DRTK rasterization and contain only
    background ``-1`` or triangle indices in ``[0, F)``. The normal-matrix CUDA
    value kernel does not validate that range because host-side validation
    would synchronize the rasterization result in this hot path. Faces must
    also have three distinct vertex indices; degenerate faces with duplicate
    corners are unsupported.

    Differentiability:
        The normal-matrix structure and rasterized triangle indices are
        discrete and non-differentiable. The returned sparse values are
        differentiable with respect to ``bary_img`` via the product rule for
        ``bary_i * bary_j``. No gradients are produced for ``vi`` or
        ``index_img``. Visibility changes remain outside this op's derivative;
        they are represented only through the supplied ``index_img``.

    Args:
        vi: Face vertex index tensor, shape ``[F, 3]``, ``[1, F, 3]``,
            or ``[N, F, 3]``. Singleton 3D batches are broadcast to ``N``.
        index_img: Rasterized triangle index image, shape ``[N, H, W]``.
        bary_img: Barycentric image, shape ``[N, 3, H, W]``.
        num_vertices: Number of vertex unknowns, i.e. the normal-matrix size.

    Returns:
        A sparse CSR tensor with shape ``[num_vertices, num_vertices]``.
    """
    if vi.ndim == 2:
        vi = vi[None].expand(index_img.shape[0], -1, -1)
    elif vi.ndim == 3 and vi.shape[0] == 1 and index_img.shape[0] != 1:
        vi = vi.expand(index_img.shape[0], -1, -1)

    crow_indices, col_indices, values = (
        th.ops.interpolate_ext.interpolation_normal_matrix(
            vi,
            index_img,
            bary_img,
            int(num_vertices),
        )
    )
    return th.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=(int(num_vertices), int(num_vertices)),
        device=values.device,
        dtype=values.dtype,
        check_invariants=False,
    )


def interpolate_ref(
    vert_attributes: th.Tensor,
    vi: th.Tensor,
    index_img: th.Tensor,
    bary_img: th.Tensor,
) -> th.Tensor:
    """Pure PyTorch reference implementation used by tests.

    This helper is intentionally not part of the documented public API. See
    :func:`drtk.interpolate` for the supported implementation.
    """

    # Run reference implementation in double precision to get as good reference as possible
    orig_dtype = vert_attributes.dtype
    vert_attributes = vert_attributes.double()
    bary_img = bary_img.double()
    b = vert_attributes.shape[0]
    iimg_clamped = index_img.clamp(min=0).long()
    vi_img = vi[iimg_clamped].long()

    v_img = th.gather(
        vert_attributes,
        1,
        vi_img.view(b, -1, 1).expand(-1, -1, vert_attributes.shape[-1]),
    )
    v_img = (
        v_img.view(*vi_img.shape[:3], 3, vert_attributes.shape[-1])
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    v_img = (v_img * bary_img[..., None]).sum(dim=1)

    # Do the sweep of value in the range -1..1 for the `index_img == -1` region, like
    # in is done in the CUDA kernel.
    undefined_region = th.stack(
        [
            (
                th.arange(0, index_img.shape[-1], device=vert_attributes.device)[
                    None, ...
                ]
                .repeat(index_img.shape[-2], 1)
                .double()
                * 2.0
                + 1.0
            )
            / index_img.shape[-1]
            - 1.0,
            (
                th.arange(0, index_img.shape[-2], device=vert_attributes.device)[
                    ..., None
                ]
                .repeat(1, index_img.shape[-1])
                .double()
                * 2.0
                + 1.0
            )
            / index_img.shape[-2]
            - 1.0,
        ],
        dim=2,
    )
    undefined_region = th.tile(
        undefined_region[None], dims=[1, 1, 1, (vert_attributes.shape[-1] + 1) // 2]
    )[:, :, :, : vert_attributes.shape[-1]]
    v_img[index_img == -1] = undefined_region.expand(index_img.shape[0], -1, -1, -1)[
        index_img == -1, :
    ]

    return v_img.permute(0, 3, 1, 2).to(orig_dtype)

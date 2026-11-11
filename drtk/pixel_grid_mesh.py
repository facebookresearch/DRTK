# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Hard pixel-cell meshes and Bernoulli expected-render-loss gradients.

Geometry is differentiable conditional on the sampled topology. Integer indices
are not differentiable; use ``bernoulli_render_loss`` around mesh/render/loss to
train topology logits. See the pixel-grid-mesh design for the exact contracts.
"""

from typing import Callable, cast, List, Optional, Tuple

import torch as th
import torch.nn.functional as thf
from drtk.utils import load_torch_ops


load_torch_ops("drtk.pixel_grid_mesh_ext")

_Build = Callable[
    [
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        bool,
    ],
    List[th.Tensor],
]
_build = cast(_Build, getattr(th.ops.pixel_grid_mesh_ext, "build", None))


PixelGridMesh = Tuple[
    th.Tensor,
    th.Tensor,
    th.Tensor,
    th.Tensor,
    Optional[th.Tensor],
    Optional[th.Tensor],
    Optional[th.Tensor],
]

_FACE_SLOTS = 32


def _realize_connections(
    alive: th.Tensor, join_right: th.Tensor, join_down: th.Tensor
) -> Tuple[th.Tensor, th.Tensor]:
    right_live = alive[:, :-1] & alive[:, 1:]
    down_live = alive[:-1] & alive[1:]
    cut_r = right_live & ~join_right
    cut_d = down_live & ~join_down
    h, w = alive.shape
    degree = th.zeros((h + 1, w + 1), dtype=th.int32, device=alive.device)
    degree[:-1, 1:-1] += cut_r
    degree[1:, 1:-1] += cut_r
    degree[1:-1, :-1] += cut_d
    degree[1:-1, 1:] += cut_d

    live = alive.to(th.int32)
    incident = (
        thf.pad(live, (0, 1, 0, 1))
        + thf.pad(live, (1, 0, 0, 1))
        + thf.pad(live, (0, 1, 1, 0))
        + thf.pad(live, (1, 0, 1, 0))
    )
    boundary = (incident > 0) & (incident < 4)
    supported = (degree > 1) | boundary
    cut_r = cut_r & (supported[:-1, 1:-1] | supported[1:, 1:-1])
    cut_d = cut_d & (supported[1:-1, :-1] | supported[1:-1, 1:])
    return right_live & ~cut_r, down_live & ~cut_d


def _corner_labels(
    h: int, w: int, join_right: th.Tensor, join_down: th.Tensor
) -> th.Tensor:
    corners = th.arange(h * w * 4, device=join_right.device).reshape(h, w, 4)
    # TL, TR, BR, BL. Links never leave their original lattice corner.
    a = th.cat(
        (
            corners[:, :-1, 1].flatten(),
            corners[:, :-1, 2].flatten(),
            corners[:-1, :, 3].flatten(),
            corners[:-1, :, 2].flatten(),
        )
    )
    b = th.cat(
        (
            corners[:, 1:, 0].flatten(),
            corners[:, 1:, 3].flatten(),
            corners[1:, :, 0].flatten(),
            corners[1:, :, 1].flatten(),
        )
    )
    joined = th.cat(
        (
            join_right.flatten(),
            join_right.flatten(),
            join_down.flatten(),
            join_down.flatten(),
        )
    )
    labels = corners.flatten()
    # A local fan contains at most four corners, hence at most three hops.
    for _ in range(3):
        next_labels = labels.clone()
        next_labels.scatter_reduce_(
            0, a, th.where(joined, labels[b], labels[a]), "amin"
        )
        next_labels.scatter_reduce_(
            0, b, th.where(joined, labels[a], labels[b]), "amin"
        )
        labels = next_labels
    return labels


def _project_pixels(v: th.Tensor, intrinsics: th.Tensor) -> th.Tensor:
    homogeneous = v @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _feather_mesh(
    v: th.Tensor,
    uv: th.Tensor,
    faces: th.Tensor,
    cells: th.Tensor,
    face_ids: th.Tensor,
    cell_corners: th.Tensor,
    alive: th.Tensor,
    right: th.Tensor,
    down: th.Tensor,
    width: th.Tensor,
    intrinsics: th.Tensor,
    inv_intrinsics: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    h, w = alive.shape
    boundary = (
        th.stack(
            (
                ~thf.pad(down, (0, 0, 1, 0)),
                ~thf.pad(right, (0, 1, 0, 0)),
                ~thf.pad(down, (0, 0, 0, 1)),
                ~thf.pad(right, (1, 0, 0, 0)),
            ),
            dim=-1,
        )
        & alive[..., None]
    )
    halfedges = boundary.flatten().nonzero().flatten()
    cell, side = halfedges // 4, halfedges % 4
    start = cell_corners[cell, side]
    end = cell_corners[cell, (side + 1) % 4]
    count = start.numel()
    edge = th.arange(count, device=v.device)
    outgoing = th.full((v.shape[0],), -1, dtype=th.long, device=v.device)
    incoming = th.full_like(outgoing, -1)
    outgoing[start] = edge
    incoming[end] = edge
    previous = incoming[start]
    following = outgoing[end]

    pixels = _project_pixels(v, intrinsics)
    tangent = pixels[end] - pixels[start]
    tiny = th.finfo(v.dtype).eps
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(tiny)
    normal = th.stack((tangent[:, 1], -tangent[:, 0]), dim=-1)
    previous_normal = normal[previous]
    vertex_width = 0.5 * (width.flatten()[cell] + width.flatten()[cell[previous]])
    bisector = normal + previous_normal
    denominator = 1.0 + (normal * previous_normal).sum(-1)
    miter = bisector / denominator.clamp_min(tiny)[:, None]
    bevel = (miter.norm(dim=-1) > 4.0) | (denominator < 1e-4)
    offset_start = th.where(bevel[:, None], normal, miter) * vertex_width[:, None]
    offset_end = th.where(
        bevel[following, None],
        normal * vertex_width[following, None],
        offset_start[following],
    )

    def extrude(anchor: th.Tensor, offset: th.Tensor) -> th.Tensor:
        screen_delta = th.cat((offset, th.zeros_like(offset[:, :1])), dim=-1)
        return v[anchor] + (screen_delta @ inv_intrinsics.T) * v[anchor, 2:3]

    outer_start = extrude(start, offset_start)
    outer_end = extrude(end, offset_end)
    first = v.shape[0] + edge
    last = th.where(bevel[following], v.shape[0] + count + edge, first[following])
    strip = th.stack(
        (th.stack((start, last, end), -1), th.stack((start, first, last), -1)),
        dim=1,
    ).reshape(-1, 3)

    # A reversal needs a cap with a third outer point, rather than a zero-area
    # bevel triangle. All outer vertices have zero material-opacity multiplier.
    reversal = denominator < 1e-4
    capped = (bevel & reversal).nonzero().flatten()
    beveled = (bevel & ~reversal).nonzero().flatten()
    cap = extrude(start[capped], tangent[previous[capped]] * vertex_width[capped, None])
    cap_id = v.shape[0] + 2 * count + th.arange(capped.numel(), device=v.device)
    bevel_faces = th.stack(
        (start[beveled], last[previous[beveled]], first[beveled]), -1
    )
    cap_faces = th.stack(
        (
            th.stack((start[capped], last[previous[capped]], cap_id), -1),
            th.stack((start[capped], cap_id, first[capped]), -1),
        ),
        dim=1,
    ).reshape(-1, 3)
    strip_ids = cell * _FACE_SLOTS + 2 + side * 6
    all_faces = th.cat((faces, strip, bevel_faces, cap_faces))
    all_cells = th.cat(
        (
            cells,
            cell.repeat_interleave(2),
            cell[beveled],
            cell[capped].repeat_interleave(2),
        )
    )
    all_ids = th.cat(
        (
            face_ids,
            (strip_ids[:, None] + th.arange(2, device=v.device)).flatten(),
            strip_ids[beveled] + 2,
            (strip_ids[capped, None] + th.tensor([3, 4], device=v.device)).flatten(),
        )
    )
    all_v = th.cat((v, outer_start, outer_end, cap))
    all_uv = th.cat((uv, uv[start], uv[end], uv[start[capped]]))
    opacity = th.cat((v.new_ones(v.shape[0]), v.new_zeros(all_v.shape[0] - v.shape[0])))
    order = all_ids.argsort()
    all_faces, all_cells, all_ids = all_faces[order], all_cells[order], all_ids[order]
    used, inverse = th.unique(all_faces.flatten(), sorted=True, return_inverse=True)
    return (
        all_v[used],
        all_uv[used],
        inverse.reshape(-1, 3),
        opacity[used],
        all_cells,
        all_ids,
    )


def pixel_grid_mesh_ref(
    depth_z: th.Tensor,
    displacement_xyz: th.Tensor,
    intrinsics: th.Tensor,
    alive: th.Tensor,
    join_right: th.Tensor,
    join_down: th.Tensor,
    diagonal: Optional[th.Tensor] = None,
    feather_width: Optional[th.Tensor] = None,
    *,
    return_metadata: bool = False,
) -> PixelGridMesh:
    """Tensor reference for :func:`pixel_grid_mesh`, supporting CPU and CUDA.

    See :func:`pixel_grid_mesh` for the complete argument and return-value
    contract.
    """
    if depth_z.ndim != 2 or min(depth_z.shape) < 1:
        raise ValueError("depth_z must have nonempty shape [H, W]")
    h, w = depth_z.shape
    expected = (
        (displacement_xyz, (h, w, 3)),
        (intrinsics, (3, 3)),
        (alive, (h, w)),
        (join_right, (h, w - 1)),
        (join_down, (h - 1, w)),
    )
    for tensor, shape in expected:
        if tuple(tensor.shape) != shape or tensor.device != depth_z.device:
            raise ValueError(
                f"expected shape {shape} on {depth_z.device}, got {tensor.shape} on {tensor.device}"
            )
    if depth_z.dtype not in (th.float32, th.float64):
        raise ValueError("depth_z must be float32 or float64")
    if displacement_xyz.dtype != depth_z.dtype or intrinsics.dtype != depth_z.dtype:
        raise ValueError("floating geometry inputs must share dtype")
    if any(t.dtype != th.bool for t in (alive, join_right, join_down)):
        raise ValueError("topology inputs must be boolean hard decisions")
    for optional, dtype in ((diagonal, th.bool), (feather_width, depth_z.dtype)):
        if optional is not None and (
            tuple(optional.shape) != (h, w)
            or optional.dtype != dtype
            or optional.device != depth_z.device
        ):
            raise ValueError(
                "diagonal/feather_width must match grid shape, dtype, and device"
            )

    right, down = _realize_connections(alive, join_right, join_down)
    labels = _corner_labels(h, w, right, down)
    y, x = th.meshgrid(
        th.arange(h, dtype=depth_z.dtype, device=depth_z.device),
        th.arange(w, dtype=depth_z.dtype, device=depth_z.device),
        indexing="ij",
    )
    centers = th.stack((x, y), -1)
    offsets = depth_z.new_tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    corner_xy = centers[..., None, :] + offsets
    homogeneous = th.cat((corner_xy, th.ones_like(corner_xy[..., :1])), -1)
    inv_intrinsics = th.linalg.inv(intrinsics)
    rays = homogeneous @ inv_intrinsics.T
    proposals = depth_z[..., None, None] * rays + displacement_xyz[..., None, :]
    active = alive.flatten().nonzero().flatten()
    representatives, inverse = th.unique(
        labels.reshape(-1, 4)[active], sorted=True, return_inverse=True
    )
    inverse = inverse.flatten()
    sums = depth_z.new_zeros((representatives.numel(), 3)).index_add(
        0, inverse, proposals.reshape(-1, 4, 3)[active].reshape(-1, 3)
    )
    counts = depth_z.new_zeros(representatives.numel()).index_add(
        0, inverse, depth_z.new_ones(inverse.numel())
    )
    v = sums / counts[:, None]
    vt0 = (corner_xy.reshape(-1, 2)[representatives] + 0.5) / depth_z.new_tensor([w, h])
    cell_corners = th.full((h * w, 4), -1, dtype=th.long, device=depth_z.device)
    cell_corners[active] = inverse.reshape(-1, 4)
    corners = cell_corners[active]
    if diagonal is None:
        center_ray = (
            th.cat((centers, th.ones_like(centers[..., :1])), -1) @ inv_intrinsics.T
        )
        target = (depth_z[..., None] * center_ray + displacement_xyz).reshape(-1, 3)[
            active
        ]
        error02 = (
            ((v[corners[:, 0]] + v[corners[:, 2]]) * 0.5 - target).square().sum(-1)
        )
        error13 = (
            ((v[corners[:, 1]] + v[corners[:, 3]]) * 0.5 - target).square().sum(-1)
        )
        chosen = error13 < error02
    else:
        chosen = diagonal.flatten()[active]
    tri02 = corners[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 2, 3)
    tri13 = corners[:, [0, 1, 3, 1, 2, 3]].reshape(-1, 2, 3)
    faces = th.where(chosen[:, None, None], tri13, tri02).reshape(-1, 3)
    cells = active.repeat_interleave(2)
    face_ids = (active[:, None] * _FACE_SLOTS + th.arange(2, device=v.device)).flatten()
    opacity: Optional[th.Tensor] = None
    if feather_width is not None:
        v, vt0, faces, opacity, cells, face_ids = _feather_mesh(
            v,
            vt0,
            faces,
            cells,
            face_ids,
            cell_corners,
            alive,
            right,
            down,
            feather_width,
            intrinsics,
            inv_intrinsics,
        )
    vt1 = (_project_pixels(v, intrinsics) + 0.5) / depth_z.new_tensor([w, h])
    return (
        v,
        vt0,
        vt1,
        faces.to(th.int32),
        opacity,
        cells if return_metadata else None,
        face_ids if return_metadata else None,
    )


def pixel_grid_mesh(
    depth_z: th.Tensor,
    displacement_xyz: th.Tensor,
    intrinsics: th.Tensor,
    alive: th.Tensor,
    join_right: th.Tensor,
    join_down: th.Tensor,
    diagonal: Optional[th.Tensor] = None,
    feather_width: Optional[th.Tensor] = None,
    *,
    return_metadata: bool = False,
) -> PixelGridMesh:
    """Build a displaced pixel-cell mesh with optional cuts and feathered borders.

    Supports float32/float64 CUDA tensors and first-order gradients.

    Args:
        depth_z: Positive camera Z depth ``[H, W]`` (float32 or float64).
        displacement_xyz: Camera-space translation proposals ``[H, W, 3]``.
            Shared corners are least-squares averages of incident proposals;
            a cell's final center need not equal its proposed center exactly.
        intrinsics: Nonsingular pinhole matrix ``[3, 3]`` with last row 0,0,1.
            Pixel centers have integer coordinates, corners half-integers.
        alive: Boolean cell participation ``[H, W]``.
        join_right: Boolean horizontal adjacency ``[H, W-1]``.
        join_down: Boolean vertical adjacency ``[H-1, W]``. Isolated interior
            one-edge cuts are healed; cuts touching a live/dead or grid border
            survive. Adjacent cut edges are evaluated before any healing.
        diagonal: Optional boolean ``[H, W]``: False selects TL--BR, True
            TR--BL. None selects the diagonal whose midpoint best matches the
            proposed cell center; ties select TL--BR.
        feather_width: Optional nonnegative widths ``[H, W]`` in source pixels.
            Generates boundary strips and fixed vertex opacity, not a learned
            opacity head. Widths are averaged within each boundary fan. Large
            widths can cause overlap; there is no global collision resolution.
        return_metadata: Return source-cell ownership and stable face-slot IDs.

    Returns:
        A fixed seven-element tuple ``(v, vt0, vt1, vi, vertex_opacity,
        face_cell_ids, face_ids)``. Vertices already include displacement and
        optional strips, in source-camera 3D, shape ``[V, 3]``. UVs have shape
        ``[V, 2]`` and faces ``[F, 3]`` int32. vt0 is the source anchor; vt1
        is the displaced source projection. Vertex opacity is ``[V]`` or None
        without feathering. The two metadata tensors are ``[F]`` int64 or
        None. No optional tuple slot is omitted.

    All tensors must share device; floating inputs must share dtype. Inputs
    must be finite, depths/widths valid, and displaced vertices in front of the
    source camera. CUDA graph capture is unsupported.

    See :func:`pixel_grid_mesh_ref` for the tensor reference implementation.
    Vertex/face storage order may differ; use stable face IDs when comparing
    geometry.
    """
    outputs = _build(
        depth_z.contiguous(),
        displacement_xyz.contiguous(),
        intrinsics.contiguous(),
        alive.contiguous(),
        join_right.contiguous(),
        join_down.contiguous(),
        th.empty(0, dtype=th.bool, device=depth_z.device)
        if diagonal is None
        else diagonal.contiguous(),
        depth_z.new_empty(0) if feather_width is None else feather_width.contiguous(),
        return_metadata,
    )
    return (
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4] if feather_width is not None else None,
        outputs[5] if return_metadata else None,
        outputs[6] if return_metadata else None,
    )


def bernoulli_topology_samples(
    logits: th.Tensor, *, generator: Optional[th.Generator] = None
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Sample two correlated hard topologies for the ARM estimator.

    Args:
        logits: Floating-point Bernoulli logits of any shape. Each element
            describes one independent proposal decision.
        generator: Optional random generator matching the logits' device.

    Returns:
        A three-element tuple ``(plus, minus, uniform)`` with the same shape
        and device as ``logits``. For ``p = sigmoid(logits)`` and shared
        ``u = uniform``, ``plus = (u > 1 - p)`` and ``minus = (u < p)`` are
        boolean tensors. ``uniform`` has the logits' floating dtype.

    Both hard samples have Bernoulli marginals with probability ``p``; they
    are correlated, not independent samples or elementwise complements.
    The returned samples and random numbers have no gradient to ``logits``.
    Use :func:`bernoulli_render_loss` to obtain an ARM loss gradient, or use
    the pair and shared noise to implement a custom estimator.
    """
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    uniform = th.rand(
        logits.shape, dtype=logits.dtype, device=logits.device, generator=generator
    )
    probability = logits.detach().sigmoid()
    return uniform > (1.0 - probability), uniform < probability, uniform


def bernoulli_render_loss(
    logits: th.Tensor,
    evaluate: Callable[[th.Tensor], th.Tensor],
    *,
    pairs: int = 1,
    generator: Optional[th.Generator] = None,
) -> th.Tensor:
    """Estimate expected rendered loss and its Bernoulli-logit gradient with ARM.

    Args:
        logits: Floating-point Bernoulli logits of any shape, typically a
            learnable tensor or the output of a prediction network.
        evaluate: Callable accepting a boolean tensor with the same shape
            as ``logits`` and returning a scalar loss tensor. It constructs
            a hard mesh, renders it, and compares the render with the target.
            Continuous parameters captured by the callable retain ordinary
            autograd through the sampled mesh and renderer.
        pairs: Positive number of correlated sample pairs. Each pair calls
            ``evaluate`` twice; more pairs reduce sampling variance at the
            cost of more evaluations.
        generator: Optional random generator matching the logits' device.

    Returns:
        A scalar whose forward value is the mean of the ``2 * pairs`` loss
        evaluations. Backward adds the ARM estimator for ``logits`` and
        preserves ordinary continuous-input gradients from those evaluations.
        Calling ``backward()`` accumulates these gradients normally.

    The score term uses detached loss differences. Any topology cleanup must
    depend on sampled bits, not detached logits. Use deterministic evaluation,
    common auxiliary randomness within each pair, or auxiliary randomness
    independent of the sampled topology.

    This helper estimates the expectation of the full loss: average losses,
    not rendered images. It does not compute visibility masks or supply
    task-specific regularizers. :func:`bernoulli_topology_samples` exposes
    the underlying correlated samples for custom estimators.
    """
    if pairs < 1:
        raise ValueError("pairs must be positive")
    result = logits.new_zeros(())
    for _ in range(pairs):
        plus, minus, uniform = bernoulli_topology_samples(logits, generator=generator)
        loss_plus, loss_minus = evaluate(plus), evaluate(minus)
        if loss_plus.ndim != 0 or loss_minus.ndim != 0:
            raise ValueError("evaluate must return a scalar loss")
        score = ((loss_plus - loss_minus).detach() * (uniform - 0.5) * logits).sum()
        result = result + 0.5 * (loss_plus + loss_minus) + (score - score.detach())
    return result / pairs

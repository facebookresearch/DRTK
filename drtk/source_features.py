# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Optional, Tuple

import torch as th
from drtk.utils import load_torch_ops

load_torch_ops("drtk.transparent_render_ext")


@th.compiler.disable
def source_transmittance(
    v_pix: th.Tensor,
    vi: th.Tensor,
    query_face_ids: th.Tensor,
    query_barycentrics: th.Tensor,
    height: int,
    width: int,
    *,
    vertex_opacity: Optional[th.Tensor] = None,
    edge_gradients: bool = True,
    return_queries: bool = False,
) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
    r"""Evaluate transmittance from a source camera or light to mesh points.

    Return bilinear image indices and weights for each point. Summing its
    four weights gives filtered transmittance, including the valid sampling
    footprint. Under unit illumination, this is the shadow intensity on the
    receiving surface.

    Each query is attached to a face by 3D barycentric coordinates. Its four
    image taps follow the receiving surface through shared manifold edges,
    stopping at cuts, projected folds, or nonmanifold junctions. Each valid
    tap is weighted by transmittance from faces preceding its receiver in
    ``(mean triangle camera Z, face index)`` order.

    Outputs retain the query order. For queries attached to an original
    surface grid, reshape the outputs into that grid after deformation.
    Queries projecting to the same camera pixel can have different
    transmittances when they lie on different surface layers.

    Apply the result to a feature image with :func:`sample_with_transmittance`.
    Feature maps with matching batch and spatial dimensions can share the
    same indices and weights. Recompute them when geometry, topology, material
    opacity, or the source camera changes.

    Args:
        v_pix (Tensor): Float32 CUDA vertices, ``(N, V, 3)``. XY are source
            pixel coordinates, with integer values at pixel centers. Z is
            positive camera depth. Use the feature image's pixel grid.
        vi (Tensor): Int32 CUDA faces, ``(F, 3)`` or ``(N, F, 3)``. Faces with
            invalid indices, nonfinite vertices, nonpositive depth, or zero
            projected area are ignored. Shared indices define connectivity;
            coincident vertices with separate indices remain separate.
        query_face_ids (Tensor): Int32 CUDA receiving-face IDs, ``(N, Q)``.
            ``-1`` denotes an unused query. At a shared vertex, choose an
            incident face on the intended side of the surface.
        query_barycentrics (Tensor): Float32 CUDA 3D barycentrics, ``(N, Q, 3)``.
            Entries must be nonnegative and sum to one. Invalid queries have
            zero weights and diagnostics, with tap indices set to ``-1``.
        height (int): Source feature-image height.
        width (int): Source feature-image width.
        vertex_opacity (Optional[Tensor]): Float32 CUDA material opacity,
            ``(N, V)``, clamped to ``[0, 1]``. Defaults to one. The receiving
            face's own opacity is excluded from visibility. Opacity must be
            available before sampling; predicting it from these same sampled
            features creates a circular dependency.
        edge_gradients (bool): Estimate geometry gradients at visibility and
            receiving-surface boundaries. Default: True.
        return_queries (bool): Return each query's projected UV and camera Z.
            Default: False.

    Returns:
        Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        ``(tap_indices, tap_weights, query_uv, query_depth)``.

        ``tap_indices`` is int32 ``(N, Q, 4)``, ordered top-left, top-right,
        bottom-left, bottom-right. Each entry is flattened as ``y * width + x``.
        Out-of-image and unused-query taps have index ``-1``. In-image indices
        remain valid at zero weight to support boundary gradients.

        ``tap_weights`` is float32 ``(N, Q, 4)``: bilinear coefficients times
        source transmittance. Taps outside the receiving surface have zero
        weight. The weights are unnormalized.

        ``query_uv`` is ``(N, Q, 2)`` in normalized ``align_corners=False``
        coordinates; ``query_depth`` is ``(N, Q)`` camera Z. Both are None
        unless requested. Unused queries have zero diagnostics.

    Gradients:
        Smooth gradients reach vertices, query barycentrics, and material
        opacity. The opacity clamp uses the interior derivative at zero and
        one. Coverage boundaries use DRTK's edge-motion estimator.
        Ordinary tessellation edges and zero-opacity occluders have zero
        boundary jump. An additional :func:`edge_grad_estimator` call would
        count this contribution twice.

        Face IDs, topology, and ordering are discrete. To optimize topology
        logits, use :func:`bernoulli_render_loss` with fresh transmittance weights
        for each sampled mesh. Only first derivatives are supported. Backward may be
        nondeterministic.

    Note:
        Geometry and weights remain float32 under autocast. CUDA graph capture
        is unsupported. :func:`source_transmittance_ref` supports CPU inputs
        and smooth derivatives.

        Visibility weights describe geometric access to the source image.
        Features mixed by transparency or a feature extractor's receptive
        field remain mixed.

    Example::

        indices, weights, _, _ = source_transmittance(
            v_pix, faces, query_faces, query_barycentrics, H, W
        )
        sampled, confidence = sample_with_transmittance(features, indices, weights)
        prediction = model(anchor_features, sampled, confidence)
        loss_fn(prediction, target).backward()
    """
    vi, opacity = _prepare(
        v_pix,
        vi,
        query_face_ids,
        query_barycentrics,
        height,
        width,
        vertex_opacity,
        native=True,
    )
    indices, weights, uv, depth = th.ops.source_features_ext.stencil(
        v_pix.contiguous(),
        vi,
        query_face_ids.contiguous(),
        query_barycentrics.contiguous(),
        opacity,
        height,
        width,
        edge_gradients,
    )
    return (
        indices,
        weights,
        uv if return_queries else None,
        depth if return_queries else None,
    )


@th.compiler.disable
def sample_with_transmittance(
    image: th.Tensor, tap_indices: th.Tensor, tap_weights: th.Tensor
) -> Tuple[th.Tensor, th.Tensor]:
    r"""Sample an image at surface points, attenuated by source transmittance.

    Use the indices and weights returned by :func:`source_transmittance`.
    For image values :math:`F_j` and weights :math:`w_{qj}`, returns
    :math:`S_q=\sum_j w_{qj}F_j` and transmittance
    :math:`c_q=\sum_j w_{qj}`. The weighted sum retains attenuation: a unit
    light image returns transmitted illumination. For feature images,
    transmittance can serve as confidence. For example,
    ``sampled + (1 - confidence) * anchor`` blends the observations with an
    existing surface feature where visibility is incomplete.

    Args:
        image (Tensor): CUDA ``(N, C, H, W)``, float16, bfloat16, or
            float32. Arbitrary nonnegative strides, including channels-last
            and sliced images, are supported. The grid and batch must match
            the weights.
        tap_indices (Tensor): Int32 CUDA ``(N, Q, 4)`` from
            :func:`source_transmittance`. ``-1`` and out-of-range indices
            contribute zero. Indices have no gradient.
        tap_weights (Tensor): Float32 CUDA ``(N, Q, 4)`` from
            :func:`source_transmittance`.

    Returns:
        Tuple[Tensor, Tensor]: ``(sampled, transmittance)``. Samples have
        shape ``(N, C, Q)`` and the image dtype, with float32 accumulation.
        Transmittance is float32 ``(N, 1, Q)``. Gradients reach the image and
        weights, including zero weights with valid indices.

    Note:
        Only first derivatives are supported. Backward may be nondeterministic.
        :func:`sample_with_transmittance_ref` supports CPU and float64 inputs.
    """
    _check_sampler(image, tap_indices, tap_weights, native=True)
    return tuple(
        th.ops.source_features_ext.sample(
            image, tap_indices.contiguous(), tap_weights.contiguous()
        )
    )


def _prepare(
    v: th.Tensor,
    vi: th.Tensor,
    qf: th.Tensor,
    qb: th.Tensor,
    height: int,
    width: int,
    opacity: Optional[th.Tensor],
    *,
    native: bool,
) -> Tuple[th.Tensor, th.Tensor]:
    if v.ndim != 3 or v.shape[-1] != 3:
        raise ValueError("v_pix must have shape (N, V, 3)")
    if vi.ndim == 2:
        vi = vi.unsqueeze(0).expand(v.shape[0], -1, -1)
    if vi.ndim != 3 or vi.shape[0] != v.shape[0] or vi.shape[-1] != 3:
        raise ValueError("vi must have shape (F, 3) or (N, F, 3)")
    if qf.ndim != 2 or qf.shape[0] != v.shape[0] or qb.shape != (*qf.shape, 3):
        raise ValueError("queries must have shapes (N, Q) and (N, Q, 3)")
    if height <= 0 or width <= 0 or height * width >= 2**31:
        raise ValueError("height and width must be positive with H*W < 2**31")
    if opacity is None:
        opacity = v.new_ones(v.shape[:2])
    if opacity.shape != v.shape[:2]:
        raise ValueError("vertex_opacity must have shape (N, V)")
    if vi.dtype != th.int32 or qf.dtype != th.int32:
        raise ValueError("vi and query_face_ids must have int32 dtype")
    if any(x.device != v.device for x in (vi, qf, qb, opacity)):
        raise ValueError("source transmittance inputs must share a device")
    if native and (
        v.device.type != "cuda" or any(x.dtype != th.float32 for x in (v, qb, opacity))
    ):
        raise ValueError(
            "native source geometry, barycentrics, opacity must be CUDA float32"
        )
    if not v.is_floating_point() or qb.dtype != v.dtype or opacity.dtype != v.dtype:
        raise ValueError(
            "geometry, barycentrics, and opacity must share a floating dtype"
        )
    return vi.contiguous(), opacity.contiguous()


def _check_sampler(
    f: th.Tensor, indices: th.Tensor, weights: th.Tensor, *, native: bool
) -> None:
    if f.ndim != 4 or indices.ndim != 3 or indices.shape[-1] != 4:
        raise ValueError("expected feature image (N,C,H,W) and indices (N,Q,4)")
    if weights.shape != indices.shape or f.shape[0] != indices.shape[0]:
        raise ValueError("sampling indices, weights, and image batch must match")
    if f.shape[2] <= 0 or f.shape[3] <= 0:
        raise ValueError("feature image spatial dimensions must be positive")
    if indices.dtype != th.int32 or any(
        x.device != f.device for x in (indices, weights)
    ):
        raise ValueError(
            "indices must be int32; all sampler inputs must share a device"
        )
    if native and (
        f.device.type != "cuda"
        or weights.dtype != th.float32
        or f.dtype not in (th.float16, th.bfloat16, th.float32)
    ):
        raise ValueError(
            "native sampler needs CUDA fp16/bf16/fp32 features and fp32 weights"
        )


def _cross(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _screen_bary(tri: th.Tensor, p: th.Tensor) -> th.Tensor:
    e1, e2, d = tri[1, :2] - tri[0, :2], tri[2, :2] - tri[0, :2], p - tri[0, :2]
    area = _cross(e1, e2)
    b1, b2 = _cross(d, e2) / area, _cross(e1, d) / area
    return th.stack((1 - b1 - b2, b1, b2))


def _covered(tri: th.Tensor, p: th.Tensor) -> bool:
    b = _screen_bary(tri, p)
    if bool((b < 0).any()):
        return False
    orientation = float(_cross(tri[1, :2] - tri[0, :2], tri[2, :2] - tri[0, :2])) > 0
    for c in range(3):
        dx, dy = (tri[(c + 2) % 3, :2] - tri[(c + 1) % 3, :2]).tolist()
        top_left = (
            (dy < 0 or (dy == 0 and dx > 0))
            if orientation
            else (dy > 0 or (dy == 0 and dx < 0))
        )
        if float(b[c]) == 0 and not top_left:
            return False
    return True


def _valid_triangle(v: th.Tensor, ids: th.Tensor) -> bool:
    if bool(((ids < 0) | (ids >= v.shape[0])).any()):
        return False
    t = v.detach()[ids.long()]
    return (
        bool(th.isfinite(t).all())
        and bool((t[:, 2] > 1e-8).all())
        and abs(float(_cross(t[1, :2] - t[0, :2], t[2, :2] - t[0, :2]))) > 1e-12
    )


def _adjacency(v: th.Tensor, vi: th.Tensor) -> List[List[int]]:
    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for f, ids in enumerate(vi.tolist()):
        if not _valid_triangle(v, vi[f]):
            continue
        for e in range(3):
            a, b = sorted((ids[e], ids[(e + 1) % 3]))
            groups.setdefault((a, b), []).append((f, e))
    adjacency = [[-1] * 3 for _ in range(vi.shape[0])]
    for (a, b), incident in groups.items():
        if len(incident) != 2:
            continue
        (f, e), (g, eg) = incident
        side_f = float(
            _cross(v[b, :2] - v[a, :2], v[int(vi[f, (e + 2) % 3]), :2] - v[a, :2])
        )
        side_g = float(
            _cross(v[b, :2] - v[a, :2], v[int(vi[g, (eg + 2) % 3]), :2] - v[a, :2])
        )
        if (side_f > 0) != (side_g > 0):
            adjacency[f][e], adjacency[g][eg] = g, f
    return adjacency


def _receiver(
    v: th.Tensor,
    vi: th.Tensor,
    adjacency: List[List[int]],
    face: int,
    origin: th.Tensor,
    target: th.Tensor,
) -> int:
    previous, progress = -1, -1e-5
    for step in range(3 * vi.shape[0] + 1):
        tri = v[vi[face].long()]
        if _covered(tri, target):
            return face
        a, b = _screen_bary(tri, origin), _screen_bary(tri, target)
        if step == 0:
            # The validated query belongs to this face before projection rounding.
            a = a.clamp_min(0)
        best: Optional[Tuple[float, int, int]] = None
        for c in range(3):
            if float(b[c]) > 0:
                continue
            e = (c + 1) % 3
            neighbor = adjacency[face][e]
            if neighbor == previous:
                continue
            denom = float(a[c] - b[c])
            t = float(a[c]) / denom if denom > 0 else 1.0
            if t < progress - 1e-5 or t > 1.00001:
                continue
            candidate = (t, e, neighbor)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None or best[2] < 0:
            return -1
        progress, _, neighbor = best
        previous, face = face, neighbor
    return -1


def source_transmittance_ref(
    v_pix: th.Tensor,
    vi: th.Tensor,
    query_face_ids: th.Tensor,
    query_barycentrics: th.Tensor,
    height: int,
    width: int,
    *,
    vertex_opacity: Optional[th.Tensor] = None,
    edge_gradients: bool = False,
    return_queries: bool = False,
) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
    """Reference for :func:`source_transmittance`, with smooth gradients only."""
    if edge_gradients:
        raise ValueError("the reference implements smooth gradients only")
    vi, opacity = _prepare(
        v_pix,
        vi,
        query_face_ids,
        query_barycentrics,
        height,
        width,
        vertex_opacity,
        native=False,
    )
    n, q = query_face_ids.shape
    zero = (
        v_pix.nan_to_num().sum()
        + query_barycentrics.nan_to_num().sum()
        + opacity.nan_to_num().sum()
    ) * 0
    indices, weights, uvs, depths = [], [], [], []
    for batch in range(n):
        v, faces, op = v_pix[batch], vi[batch], opacity[batch]
        adjacency = _adjacency(v.detach(), faces)
        order = sorted(
            (float(v.detach()[ids.long(), 2].mean()), f)
            for f, ids in enumerate(faces)
            if _valid_triangle(v, ids)
        )
        rank = {f: i for i, (_, f) in enumerate(order)}
        for query in range(q):
            f = int(query_face_ids[batch, query])
            beta = query_barycentrics[batch, query]
            valid = (
                f in rank
                and bool(th.isfinite(beta).all())
                and bool((beta >= 0).all())
                and abs(float(beta.detach().sum()) - 1) < 1e-5
            )
            if not valid:
                indices.extend([-1] * 4)
                weights.extend([zero] * 4)
                uvs.extend([zero, zero])
                depths.append(zero)
                continue
            tri = v[faces[f].long()]
            z = (beta * tri[:, 2]).sum()
            p = (beta[:, None] * tri[:, 2:3] * tri[:, :2]).sum(0) / z
            uvs.extend((2 * (p[0] + 0.5) / width - 1, 2 * (p[1] + 0.5) / height - 1))
            depths.append(z)
            x0, y0 = int(th.floor(p[0])), int(th.floor(p[1]))
            fx, fy = p[0] - x0, p[1] - y0
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                x, y = x0 + dx, y0 + dy
                b = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                if x < 0 or y < 0 or x >= width or y >= height:
                    indices.append(-1)
                    weights.append(zero)
                    continue
                indices.append(y * width + x)
                tap = p.new_tensor((x, y))
                receiver = _receiver(v.detach(), faces, adjacency, f, p.detach(), tap)
                if receiver < 0:
                    weights.append(zero)
                    continue
                transmittance = zero + 1
                for _, other in order[: rank[receiver]]:
                    other_tri = v[faces[other].long()]
                    if _covered(other_tri.detach(), tap):
                        sb = _screen_bary(other_tri, tap)
                        pb = sb / other_tri[:, 2]
                        pb = pb / pb.sum()
                        material = op[faces[other].long()]
                        alpha = (
                            material[0]
                            + pb[1] * (material[1] - material[0])
                            + pb[2] * (material[2] - material[0])
                        )
                        # Use the interior derivative at both clamp endpoints.
                        alpha = th.where(
                            alpha < 0,
                            th.zeros_like(alpha),
                            th.where(alpha > 1, th.ones_like(alpha), alpha),
                        )
                        transmittance = transmittance * (1 - alpha)
                weights.append(b * transmittance)
    idx = th.tensor(indices, dtype=th.int32, device=v_pix.device).reshape(n, q, 4)
    w = th.stack(weights).reshape(n, q, 4) if weights else zero.expand(n, q, 4)
    uv = th.stack(uvs).reshape(n, q, 2) if uvs else zero.expand(n, q, 2)
    z = th.stack(depths).reshape(n, q) if depths else zero.expand(n, q)
    return idx, w, uv if return_queries else None, z if return_queries else None


def sample_with_transmittance_ref(
    image: th.Tensor, tap_indices: th.Tensor, tap_weights: th.Tensor
) -> Tuple[th.Tensor, th.Tensor]:
    """Reference for :func:`sample_with_transmittance`, supporting CPU and float64."""
    _check_sampler(image, tap_indices, tap_weights, native=False)
    n, c, h, w = image.shape
    q = tap_indices.shape[1]
    valid = (tap_indices >= 0) & (tap_indices < h * w)
    idx = (
        tap_indices.clamp(0, h * w - 1).long().reshape(n, 1, q * 4).expand(n, c, q * 4)
    )
    samples = image.reshape(n, c, h * w).gather(2, idx).reshape(n, c, q, 4)
    weights = tap_weights * valid
    return (samples * weights[:, None]).sum(-1), weights.sum(-1)[:, None]

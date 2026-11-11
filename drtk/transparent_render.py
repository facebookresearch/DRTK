# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Differentiable rendering of depth-sorted transparent triangles.

The optimized operators in this module combine triangle coverage,
perspective-correct attribute interpolation, texture/feature sampling, and
front-to-back alpha compositing.  Keeping those operations together avoids
materializing a fragment buffer whose size is proportional to overdraw.

Sorting uses mean vertex depth for each triangle. It is an approximation for
small triangles with consistent local ordering, not a per-pixel depth sort:
intersecting or depth-crossing triangles can composite in the wrong order.
Independent feature UV sets do not perform visibility/provenance gating;
callers must reject samples from unrelated surfaces before appearance shading.

The ``*_ref`` functions are intentionally simple PyTorch implementations.
They are correctness oracles for tests and small experiments, not production
renderers: their complexity is ``O(batch * faces * height * width)``.

Appearance and opacity clamps use a unit subgradient at their endpoints and
zero outside the interval. This convention is explicit in the references,
independent of the installed PyTorch build. ReLU retains zero gradient at zero,
including the lower bound on final fragment alpha.
"""

import math
from typing import Callable, cast, Dict, List, Optional, Tuple

import torch as th
import torch.nn.functional as thf
from drtk.utils import load_torch_ops


__all__ = [
    "render_transparent",
    "render_transparent_mlp",
    "render_transparent_mlp_two_textures",
    "render_transparent_ref",
    "render_transparent_mlp_ref",
    "render_transparent_mlp_two_textures_ref",
]


try:
    load_torch_ops("drtk.transparent_render_ext")
except ImportError:
    # The extension has CUDA sources only, so a ROCm build's module is empty; importing drtk
    # must still work there, and the transparent operators below are then None.
    if th.version.hip is None:
        raise


_RenderWithTexture = Callable[
    [
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        th.Tensor,
        int,
        int,
        int,
        int,
        float,
        int,
        int,
        bool,
        int,
        float,
        float,
    ],
    List[th.Tensor],
]
_RenderWithMlp = Callable[..., List[th.Tensor]]
_render_with_texture = cast(
    _RenderWithTexture,
    getattr(th.ops.transparent_render_ext, "render_with_texture", None),
)
_render_with_mlp = cast(
    _RenderWithMlp,
    getattr(th.ops.transparent_render_ext, "render_with_mlp", None),
)


_DEPTH_MODES: Dict[str, int] = {
    "expected": 0,
    "accumulated": 1,
    "foreground_mean": 2,
    "median": 3,
}


def _validate_depth_options(
    depth_mode: str, background_depth: float, depth_alpha_threshold: float
) -> int:
    if depth_mode not in _DEPTH_MODES:
        raise ValueError(f"transparent render: unknown depth_mode {depth_mode!r}")
    if (
        not math.isfinite(background_depth)
        or not 0.0 < background_depth <= 3.4028234663852886e38
    ):
        raise ValueError(
            "transparent render: background_depth must be positive and finite in float32"
        )
    if (
        not math.isfinite(depth_alpha_threshold)
        or not 1.1754943508222875e-38 <= depth_alpha_threshold < 1.0
    ):
        raise ValueError(
            "transparent render: depth_alpha_threshold must be positive and less than one in float32"
        )
    return _DEPTH_MODES[depth_mode]


def _reference_depth(
    weighted: th.Tensor,
    alpha: th.Tensor,
    transmittance: th.Tensor,
    median: th.Tensor,
    depth_mode: str,
    background_depth: float,
    depth_alpha_threshold: float,
) -> th.Tensor:
    if depth_mode == "expected":
        return weighted + transmittance * background_depth
    if depth_mode == "accumulated":
        return weighted
    if depth_mode == "median":
        return median
    valid = alpha > depth_alpha_threshold
    denominator = th.where(valid, alpha, th.ones_like(alpha))
    return th.where(
        valid, weighted / denominator, th.full_like(weighted, background_depth)
    )


_MLP_WIDTH: int = 8
_MAX_FEATURE_CHANNELS: int = 32
_SUPPORTED_MLP_LAYER_COUNTS: Tuple[int, int] = (4, 5)
_ALPHA_MAX: float = 1.0
_EXP_DENSITY_MAX: float = 15.0
_TEXTURE_RGB_ACTIVATIONS: Dict[str, int] = {
    "relu": 0,
    "identity": 1,
    "shifted_sigmoid": 2,
    "clamp": 3,
}
_TEXTURE_ALPHA_ACTIVATIONS: Dict[str, int] = {
    "relu_density": 0,
    "shifted_sigmoid": 1,
    "exp_density": 2,
    "clamp": 3,
}
_MLP_RGB_ACTIVATIONS: Dict[str, int] = {
    "shifted_relu": 0,
    "shifted_identity": 1,
    "sigmoid": 2,
    "shifted_clamp": 3,
}
_MLP_ALPHA_ACTIVATIONS: Dict[str, int] = {
    "shifted_relu_density": 0,
    "sigmoid": 1,
    "exp_density": 2,
    "shifted_clamp": 3,
}


def _epsclamp(value: th.Tensor) -> th.Tensor:
    """Clamp away from zero while preserving the sign of ``value``."""

    eps = 1.0e-12
    return th.where(value < 0, value.clamp(max=-eps), value.clamp(min=eps))


def _select_batched(value: th.Tensor, batch_index: int, batch_size: int) -> th.Tensor:
    if value.shape[0] == batch_size:
        return value[batch_index]
    if value.shape[0] == 1:
        return value[0]
    raise ValueError(
        "transparent render: expected a batch dimension of 1 or "
        f"{batch_size}, but got {value.shape[0]}"
    )


def _faces_for_batch(vi: th.Tensor, batch_index: int, batch_size: int) -> th.Tensor:
    if vi.ndim == 2:
        return vi
    if vi.ndim != 3:
        raise ValueError(
            "transparent render: vi must have shape [F, 3], [1, F, 3], or "
            f"[N, F, 3], but got {tuple(vi.shape)}"
        )
    return _select_batched(vi, batch_index, batch_size)


def _vertex_attribute_for_batch(
    value: th.Tensor,
    batch_index: int,
    batch_size: int,
    channels: int,
    name: str,
) -> th.Tensor:
    if channels == 1 and value.ndim == 1:
        return value
    if channels != 1 and value.ndim == 2:
        return value
    expected_dims = 2 if channels == 1 else 3
    if value.ndim != expected_dims:
        suffix = (
            "[V] or [N, V]"
            if channels == 1
            else f"[V, {channels}] or [N, V, {channels}]"
        )
        raise ValueError(
            f"transparent render: {name} must have shape {suffix}, but got "
            f"{tuple(value.shape)}"
        )
    return _select_batched(value, batch_index, batch_size)


def _background_for_batch(
    background: Optional[th.Tensor],
    batch_index: int,
    batch_size: int,
    like: th.Tensor,
) -> th.Tensor:
    if background is None:
        return like.new_zeros((3,))
    if background.ndim == 1:
        if background.shape[0] != 3:
            raise ValueError(
                "transparent render: background must have three channels, but got "
                f"{tuple(background.shape)}"
            )
        return background
    if background.ndim != 2 or background.shape[1] != 3:
        raise ValueError(
            "transparent render: background must have shape [3], [1, 3], or "
            f"[N, 3], but got {tuple(background.shape)}"
        )
    return _select_batched(background, batch_index, batch_size)


def _texture_for_batch(
    texture: th.Tensor,
    batch_index: int,
    batch_size: int,
    name: str,
) -> th.Tensor:
    if texture.ndim != 4:
        raise ValueError(
            f"transparent render: {name} must have shape [N, C, H, W], but got "
            f"{tuple(texture.shape)}"
        )
    return _select_batched(texture, batch_index, batch_size)[None]


def _pixel_grid(
    height: int,
    width: int,
    like: th.Tensor,
) -> th.Tensor:
    # DRTK defines pixel sample locations at integer coordinates, rather than
    # the half-integer convention used by some other rasterizers.
    y, x = th.meshgrid(
        th.arange(height, dtype=like.dtype, device=like.device),
        th.arange(width, dtype=like.dtype, device=like.device),
        indexing="ij",
    )
    return th.stack((x, y), dim=0)


def _edge_function(
    a: th.Tensor,
    b: th.Tensor,
    grid: th.Tensor,
) -> th.Tensor:
    edge = b - a
    delta = grid - a[:, None, None]
    return edge[0] * delta[1] - edge[1] * delta[0]


def _canonical_edge_function(
    index_a: th.Tensor,
    index_b: th.Tensor,
    a: th.Tensor,
    b: th.Tensor,
    grid: th.Tensor,
) -> th.Tensor:
    # Match DRTK's endpoint ordering so a shared edge produces identical
    # floating-point values for both adjacent triangles.
    if bool((index_a <= index_b).item()):
        return _edge_function(a, b, grid)
    return -_edge_function(b, a, grid)


def _triangle_fields(
    triangle: th.Tensor,
    face: th.Tensor,
    grid: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor]:
    """Return perspective-correct barycentrics and hard DRTK coverage."""

    p0 = triangle[0, :2]
    p1 = triangle[1, :2]
    p2 = triangle[2, :2]
    edge01 = p1 - p0
    edge02 = p2 - p0
    area = edge01[0] * edge02[1] - edge01[1] * edge02[0]
    edge12 = p2 - p1
    sign = th.where(area >= 0, area.new_ones(()), area.new_full((), -1.0))
    raw_bary = (
        th.stack(
            (
                _canonical_edge_function(face[1], face[2], p1, p2, grid),
                _canonical_edge_function(face[2], face[0], p2, p0, grid),
                _canonical_edge_function(face[0], face[1], p0, p1, grid),
            ),
            dim=0,
        )
        * sign
    )
    screen_bary = raw_bary / area.abs().clamp(min=1.0e-12)

    inv_z = 1.0 / _epsclamp(triangle[:, 2])
    weighted_bary = screen_bary * inv_z[:, None, None]
    bary = weighted_bary / _epsclamp(weighted_bary.sum(dim=0))[None]

    positive_winding = area > 0
    is_top_left0 = th.where(
        positive_winding,
        (edge12[1] < 0) | ((edge12[1] == 0) & (edge12[0] > 0)),
        (edge12[1] > 0) | ((edge12[1] == 0) & (edge12[0] < 0)),
    )
    is_top_left1 = th.where(
        positive_winding,
        (edge02[1] > 0) | ((edge02[1] == 0) & (edge02[0] < 0)),
        (edge02[1] < 0) | ((edge02[1] == 0) & (edge02[0] > 0)),
    )
    is_top_left2 = th.where(
        positive_winding,
        (edge01[1] < 0) | ((edge01[1] == 0) & (edge01[0] > 0)),
        (edge01[1] > 0) | ((edge01[1] == 0) & (edge01[0] < 0)),
    )
    coverage = (
        (raw_bary[0] >= 0)
        & (raw_bary[1] >= 0)
        & (raw_bary[2] >= 0)
        & ((raw_bary[0] != 0) | is_top_left0)
        & ((raw_bary[1] != 0) | is_top_left1)
        & ((raw_bary[2] != 0) | is_top_left2)
        & (area.abs() > 1.0e-12)
    ).to(triangle.dtype)
    return bary, coverage


def _interpolate_triangle_attribute(
    attribute: th.Tensor,
    face: th.Tensor,
    bary: th.Tensor,
) -> th.Tensor:
    corners = attribute[face.to(dtype=th.long)]
    if corners.ndim == 1:
        return (
            corners[0]
            + bary[1] * (corners[1] - corners[0])
            + bary[2] * (corners[2] - corners[0])
        )
    return (corners[:, :, None, None] * bary[:, None]).sum(dim=0)


def _sample_texture(
    texture: th.Tensor,
    uv: th.Tensor,
    mode: str,
    padding_mode: str,
) -> th.Tensor:
    # Public UVs use the usual graphics [0, 1] convention. grid_sample uses
    # [-1, 1], with align_corners=False mapping 0/1 to the texture boundaries.
    sample_grid = (uv.permute(1, 2, 0)[None] * 2.0) - 1.0
    return thf.grid_sample(
        texture,
        sample_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )[0]


def _validate_common(
    v: th.Tensor,
    vi: th.Tensor,
    height: int,
    width: int,
) -> None:
    if v.ndim != 3 or v.shape[2] != 3:
        raise ValueError(
            f"transparent render: v must have shape [N, V, 3], but got {tuple(v.shape)}"
        )
    if vi.shape[-1] != 3:
        raise ValueError(
            f"transparent render: vi must contain triangles, but got {tuple(vi.shape)}"
        )
    if height <= 0 or width <= 0:
        raise ValueError(
            "transparent render: height and width must be positive, but got "
            f"{height} x {width}"
        )


def _validate_sampling_options(mode: str, padding_mode: str) -> Tuple[int, int]:
    if mode not in ("bilinear", "nearest", "bicubic"):
        raise ValueError(
            "transparent render: expected mode to be 'bilinear', 'nearest', or "
            f"'bicubic', but got: '{mode}'"
        )
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError(
            "transparent render: expected padding_mode to be 'zeros', 'border', "
            f"or 'reflection', but got: '{padding_mode}'"
        )
    mode_enum = {"bilinear": 0, "nearest": 1, "bicubic": 2}[mode]
    padding_mode_enum = {"zeros": 0, "border": 1, "reflection": 2}[padding_mode]
    return mode_enum, padding_mode_enum


def _validate_texture_activation_options(
    rgb_activation: str,
    alpha_activation: str,
) -> Tuple[int, int]:
    if rgb_activation not in _TEXTURE_RGB_ACTIVATIONS:
        raise ValueError(
            "transparent render: expected rgb_activation to be 'relu', "
            f"'identity', 'shifted_sigmoid', or 'clamp', but got: "
            f"'{rgb_activation}'"
        )
    if alpha_activation not in _TEXTURE_ALPHA_ACTIVATIONS:
        raise ValueError(
            "transparent render: expected alpha_activation to be 'relu_density', "
            f"'shifted_sigmoid', 'exp_density', or 'clamp', but got: "
            f"'{alpha_activation}'"
        )
    return (
        _TEXTURE_RGB_ACTIVATIONS[rgb_activation],
        _TEXTURE_ALPHA_ACTIVATIONS[alpha_activation],
    )


def _validate_mlp_activation_options(
    rgb_activation: str,
    alpha_activation: str,
) -> Tuple[int, int]:
    if rgb_activation not in _MLP_RGB_ACTIVATIONS:
        raise ValueError(
            "transparent render MLP: expected rgb_activation to be "
            "'shifted_relu', 'shifted_identity', 'sigmoid', or "
            f"'shifted_clamp', but got: '{rgb_activation}'"
        )
    if alpha_activation not in _MLP_ALPHA_ACTIVATIONS:
        raise ValueError(
            "transparent render MLP: expected alpha_activation to be "
            "'shifted_relu_density', 'sigmoid', 'exp_density', or "
            f"'shifted_clamp', but got: '{alpha_activation}'"
        )
    return (
        _MLP_RGB_ACTIVATIONS[rgb_activation],
        _MLP_ALPHA_ACTIVATIONS[alpha_activation],
    )


def _clamp_with_inclusive_grad(
    value: th.Tensor,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> th.Tensor:
    """Clamp with the native renderer's unit subgradient at either endpoint.

    Scalar Tensor.clamp uses different endpoint derivatives in fbcode and
    upstream PyTorch. Select the input at equality to keep this oracle stable.
    """
    if min_value is not None:
        value = th.where(value < min_value, min_value, value)
    if max_value is not None:
        value = th.where(value > max_value, max_value, value)
    return value


def _activate_texture_rgb(value: th.Tensor, activation: str) -> th.Tensor:
    if activation == "relu":
        return th.relu(value)
    if activation == "identity":
        return value
    if activation == "shifted_sigmoid":
        return (value - 0.5).sigmoid()
    if activation == "clamp":
        return _clamp_with_inclusive_grad(value, 0.0, 1.0)
    raise AssertionError(f"unvalidated texture RGB activation: {activation}")


def _activate_texture_alpha(value: th.Tensor, activation: str) -> th.Tensor:
    if activation == "relu_density":
        return -th.expm1(-th.relu(value))
    if activation == "shifted_sigmoid":
        return (value - 0.5).sigmoid()
    if activation == "exp_density":
        density = _clamp_with_inclusive_grad(value, max_value=_EXP_DENSITY_MAX).exp()
        return -th.expm1(-density)
    if activation == "clamp":
        return _clamp_with_inclusive_grad(value, 0.0, 1.0)
    raise AssertionError(f"unvalidated texture alpha activation: {activation}")


def _activate_mlp_rgb(value: th.Tensor, activation: str) -> th.Tensor:
    if activation == "shifted_relu":
        return th.relu(value + 0.5)
    if activation == "shifted_identity":
        return value + 0.5
    if activation == "sigmoid":
        return value.sigmoid()
    if activation == "shifted_clamp":
        return _clamp_with_inclusive_grad(value + 0.5, 0.0, 1.0)
    raise AssertionError(f"unvalidated MLP RGB activation: {activation}")


def _activate_mlp_alpha(value: th.Tensor, activation: str) -> th.Tensor:
    if activation == "shifted_relu_density":
        return -th.expm1(-th.relu(value + 0.5))
    if activation == "sigmoid":
        return value.sigmoid()
    if activation == "exp_density":
        density = _clamp_with_inclusive_grad(value, max_value=_EXP_DENSITY_MAX).exp()
        return -th.expm1(-density)
    if activation == "shifted_clamp":
        return _clamp_with_inclusive_grad(value + 0.5, 0.0, 1.0)
    raise AssertionError(f"unvalidated MLP alpha activation: {activation}")


def _validate_mlp(
    feature_channels: int,
    weights: List[th.Tensor],
    biases: List[th.Tensor],
) -> None:
    if not 0 < feature_channels <= _MAX_FEATURE_CHANNELS:
        raise ValueError(
            "transparent render MLP: total feature channels must be in [1, "
            f"{_MAX_FEATURE_CHANNELS}], but got {feature_channels}"
        )
    layer_count = len(weights)
    if layer_count not in _SUPPORTED_MLP_LAYER_COUNTS:
        raise ValueError(
            "transparent render MLP: expected exactly 4 or 5 affine layers, but "
            f"got {layer_count}"
        )
    if len(biases) != layer_count:
        raise ValueError(
            "transparent render MLP: weights and biases must have equal lengths, "
            f"but got {layer_count} and {len(biases)}"
        )

    expected_input = feature_channels
    for layer, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        expected_output = 4 if layer == layer_count - 1 else _MLP_WIDTH
        expected_weight_shape = (expected_output, expected_input)
        if tuple(weight.shape) != expected_weight_shape:
            raise ValueError(
                f"transparent render MLP: layer {layer} weight must have shape "
                f"{expected_weight_shape}, but got {tuple(weight.shape)}"
            )
        if tuple(bias.shape) != (expected_output,):
            raise ValueError(
                f"transparent render MLP: layer {layer} bias must have shape "
                f"({expected_output},), but got {tuple(bias.shape)}"
            )
        expected_input = expected_output


def _tiny_mlp(
    features: th.Tensor,
    weights: List[th.Tensor],
    biases: List[th.Tensor],
    negative_slope: float,
    rgb_activation: str,
    alpha_activation: str,
) -> th.Tensor:
    value = features
    last_layer = len(weights) - 1
    for layer, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        value = th.einsum("oc,chw->ohw", weight, value) + bias[:, None, None]
        if layer == last_layer:
            value = th.cat(
                (
                    _activate_mlp_rgb(value[:3], rgb_activation),
                    _activate_mlp_alpha(value[3:], alpha_activation),
                ),
                dim=0,
            )
        else:
            value = thf.leaky_relu(value, negative_slope=negative_slope)
    return value


def render_transparent_ref(
    v: th.Tensor,
    vi: th.Tensor,
    uv: th.Tensor,
    texture: th.Tensor,
    height: int,
    width: int,
    opacity: Optional[th.Tensor] = None,
    background: Optional[th.Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "border",
    rgb_activation: str = "relu",
    alpha_activation: str = "relu_density",
    *,
    depth_mode: str = "expected",
    background_depth: float = 1000.0,
    depth_alpha_threshold: float = 1.0e-6,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Pure-PyTorch correctness reference for :func:`render_transparent`.

    See the main function for the argument and return-value contract.
    This CPU/CUDA reference differentiates smooth shading and compositing
    with coverage and triangle order fixed. It does not implement the
    CUDA entry point's silhouette-gradient estimator or early termination.
    """

    _validate_depth_options(depth_mode, background_depth, depth_alpha_threshold)
    _validate_common(v, vi, height, width)
    _validate_sampling_options(mode, padding_mode)
    _validate_texture_activation_options(rgb_activation, alpha_activation)
    if uv.shape[-1] != 2:
        raise ValueError(
            f"transparent render: uv must have two channels, but got {tuple(uv.shape)}"
        )
    if texture.shape[1] not in (3, 4):
        raise ValueError(
            "transparent render: texture must have three or four channels, but "
            f"got {texture.shape[1]}"
        )

    batch_size = v.shape[0]
    grid = _pixel_grid(height, width, v)
    colors: List[th.Tensor] = []
    alphas: List[th.Tensor] = []
    depths: List[th.Tensor] = []

    for batch_index in range(batch_size):
        faces = _faces_for_batch(vi, batch_index, batch_size)
        uv_batch = _vertex_attribute_for_batch(uv, batch_index, batch_size, 2, "uv")
        texture_batch = _texture_for_batch(texture, batch_index, batch_size, "texture")
        if opacity is None:
            opacity_batch = v.new_ones((v.shape[1],))
        else:
            opacity_batch = _vertex_attribute_for_batch(
                opacity, batch_index, batch_size, 1, "opacity"
            )

        face_vertices = v[batch_index][faces.to(dtype=th.long)]
        face_depth = face_vertices[:, :, 2].mean(dim=1).detach()
        order = th.argsort(face_depth, stable=True)

        transmittance = v.new_ones((height, width))
        color = v.new_zeros((3, height, width))
        accumulated_alpha = v.new_zeros((height, width))
        weighted_depth = v.new_zeros((height, width))
        median_depth = v.new_full((height, width), background_depth)
        median_selected = th.zeros((height, width), device=v.device, dtype=th.bool)
        for rank in range(faces.shape[0]):
            face_index = order[rank]
            face = faces[face_index]
            triangle = face_vertices[face_index]
            bary, coverage = _triangle_fields(triangle, face, grid)
            interpolated_uv = _interpolate_triangle_attribute(uv_batch, face, bary)
            texel = _sample_texture(texture_batch, interpolated_uv, mode, padding_mode)
            material_opacity = _interpolate_triangle_attribute(
                opacity_batch, face, bary
            )
            fragment_rgb = _activate_texture_rgb(texel[:3], rgb_activation)
            texel_alpha = (
                _activate_texture_alpha(texel[3], alpha_activation)
                if texture_batch.shape[1] == 4
                else 1.0
            )
            raw_fragment_alpha = (
                coverage
                * _clamp_with_inclusive_grad(material_opacity, 0.0, 1.0)
                * texel_alpha
            )
            fragment_alpha = _clamp_with_inclusive_grad(
                th.relu(raw_fragment_alpha), max_value=_ALPHA_MAX
            )
            visibility = transmittance * fragment_alpha
            color = color + fragment_rgb * visibility[None]
            fragment_depth = (bary * triangle[:, 2, None, None]).sum(dim=0)
            accumulated_alpha = accumulated_alpha + visibility
            if depth_mode == "median":
                selected = ~median_selected & (accumulated_alpha >= 0.5)
                median_depth = th.where(selected, fragment_depth, median_depth)
                median_selected = median_selected | selected
            else:
                weighted_depth = weighted_depth + visibility * fragment_depth
            transmittance = transmittance * (1.0 - fragment_alpha)

        background_batch = _background_for_batch(background, batch_index, batch_size, v)
        color = color + background_batch[:, None, None] * transmittance[None]
        colors.append(color)
        alphas.append(accumulated_alpha[None])
        depths.append(
            _reference_depth(
                weighted_depth,
                accumulated_alpha,
                transmittance,
                median_depth,
                depth_mode,
                background_depth,
                depth_alpha_threshold,
            )[None]
        )

    return th.stack(colors, dim=0), th.stack(alphas, dim=0), th.stack(depths, dim=0)


def render_transparent_mlp_ref(
    v: th.Tensor,
    vi: th.Tensor,
    uv: th.Tensor,
    texture: th.Tensor,
    weights: List[th.Tensor],
    biases: List[th.Tensor],
    height: int,
    width: int,
    opacity: Optional[th.Tensor] = None,
    background: Optional[th.Tensor] = None,
    negative_slope: float = 0.2,
    mode: str = "bilinear",
    padding_mode: str = "border",
    rgb_activation: str = "shifted_relu",
    alpha_activation: str = "shifted_relu_density",
    *,
    depth_mode: str = "expected",
    background_depth: float = 1000.0,
    depth_alpha_threshold: float = 1.0e-6,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Pure-PyTorch correctness reference for :func:`render_transparent_mlp`.

    See the main function for the argument and return-value contract.
    This CPU/CUDA reference differentiates smooth shading and compositing
    with coverage and triangle order fixed. It does not implement the
    CUDA entry point's silhouette-gradient estimator or early termination.
    """

    empty_texture = texture.new_empty((1, 0, 1, 1))
    return _render_transparent_mlp_ref_impl(
        v,
        vi,
        uv,
        uv,
        texture,
        empty_texture,
        weights,
        biases,
        height,
        width,
        opacity,
        background,
        negative_slope,
        mode,
        padding_mode,
        rgb_activation,
        alpha_activation,
        depth_mode=depth_mode,
        background_depth=background_depth,
        depth_alpha_threshold=depth_alpha_threshold,
    )


def render_transparent_mlp_two_textures_ref(
    v: th.Tensor,
    vi: th.Tensor,
    uv0: th.Tensor,
    uv1: th.Tensor,
    texture0: th.Tensor,
    texture1: th.Tensor,
    weights: List[th.Tensor],
    biases: List[th.Tensor],
    height: int,
    width: int,
    opacity: Optional[th.Tensor] = None,
    background: Optional[th.Tensor] = None,
    negative_slope: float = 0.2,
    mode: str = "bilinear",
    padding_mode: str = "border",
    rgb_activation: str = "shifted_relu",
    alpha_activation: str = "shifted_relu_density",
    *,
    depth_mode: str = "expected",
    background_depth: float = 1000.0,
    depth_alpha_threshold: float = 1.0e-6,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Pure-PyTorch correctness reference for :func:`render_transparent_mlp_two_textures`.

    See the main function for the argument and return-value contract.
    This CPU/CUDA reference differentiates smooth shading and compositing
    with coverage and triangle order fixed. It does not implement the
    CUDA entry point's silhouette-gradient estimator or early termination.
    """

    return _render_transparent_mlp_ref_impl(
        v,
        vi,
        uv0,
        uv1,
        texture0,
        texture1,
        weights,
        biases,
        height,
        width,
        opacity,
        background,
        negative_slope,
        mode,
        padding_mode,
        rgb_activation,
        alpha_activation,
        depth_mode=depth_mode,
        background_depth=background_depth,
        depth_alpha_threshold=depth_alpha_threshold,
    )


def _render_transparent_mlp_ref_impl(
    v: th.Tensor,
    vi: th.Tensor,
    uv0: th.Tensor,
    uv1: th.Tensor,
    texture0: th.Tensor,
    texture1: th.Tensor,
    weights: List[th.Tensor],
    biases: List[th.Tensor],
    height: int,
    width: int,
    opacity: Optional[th.Tensor] = None,
    background: Optional[th.Tensor] = None,
    negative_slope: float = 0.2,
    mode: str = "bilinear",
    padding_mode: str = "border",
    rgb_activation: str = "shifted_relu",
    alpha_activation: str = "shifted_relu_density",
    *,
    depth_mode: str = "expected",
    background_depth: float = 1000.0,
    depth_alpha_threshold: float = 1.0e-6,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Shared implementation for the one- and two-texture reference APIs."""

    _validate_depth_options(depth_mode, background_depth, depth_alpha_threshold)
    _validate_common(v, vi, height, width)
    _validate_sampling_options(mode, padding_mode)
    _validate_mlp_activation_options(rgb_activation, alpha_activation)
    if uv0.shape[-1] != 2 or uv1.shape[-1] != 2:
        raise ValueError("transparent render MLP: uv0 and uv1 must have two channels")
    feature_channels = texture0.shape[1] + texture1.shape[1]
    _validate_mlp(feature_channels, weights, biases)

    batch_size = v.shape[0]
    grid = _pixel_grid(height, width, v)
    colors: List[th.Tensor] = []
    alphas: List[th.Tensor] = []
    depths: List[th.Tensor] = []

    for batch_index in range(batch_size):
        faces = _faces_for_batch(vi, batch_index, batch_size)
        uv0_batch = _vertex_attribute_for_batch(uv0, batch_index, batch_size, 2, "uv0")
        uv1_batch = _vertex_attribute_for_batch(uv1, batch_index, batch_size, 2, "uv1")
        texture0_batch = _texture_for_batch(
            texture0, batch_index, batch_size, "texture0"
        )
        texture1_batch = _texture_for_batch(
            texture1, batch_index, batch_size, "texture1"
        )
        if opacity is None:
            opacity_batch = v.new_ones((v.shape[1],))
        else:
            opacity_batch = _vertex_attribute_for_batch(
                opacity, batch_index, batch_size, 1, "opacity"
            )

        face_vertices = v[batch_index][faces.to(dtype=th.long)]
        face_depth = face_vertices[:, :, 2].mean(dim=1).detach()
        order = th.argsort(face_depth, stable=True)

        transmittance = v.new_ones((height, width))
        color = v.new_zeros((3, height, width))
        accumulated_alpha = v.new_zeros((height, width))
        weighted_depth = v.new_zeros((height, width))
        median_depth = v.new_full((height, width), background_depth)
        median_selected = th.zeros((height, width), device=v.device, dtype=th.bool)
        for rank in range(faces.shape[0]):
            face_index = order[rank]
            face = faces[face_index]
            triangle = face_vertices[face_index]
            bary, coverage = _triangle_fields(triangle, face, grid)
            interpolated_uv0 = _interpolate_triangle_attribute(uv0_batch, face, bary)
            interpolated_uv1 = _interpolate_triangle_attribute(uv1_batch, face, bary)
            sampled_features = th.cat(
                (
                    _sample_texture(
                        texture0_batch, interpolated_uv0, mode, padding_mode
                    ),
                    _sample_texture(
                        texture1_batch,
                        interpolated_uv1,
                        mode,
                        padding_mode,
                    ),
                ),
                dim=0,
            )
            fragment_rgba = _tiny_mlp(
                sampled_features,
                weights,
                biases,
                negative_slope,
                rgb_activation,
                alpha_activation,
            )
            material_opacity = _interpolate_triangle_attribute(
                opacity_batch, face, bary
            )
            raw_fragment_alpha = (
                coverage
                * _clamp_with_inclusive_grad(material_opacity, 0.0, 1.0)
                * fragment_rgba[3]
            )
            fragment_alpha = _clamp_with_inclusive_grad(
                th.relu(raw_fragment_alpha), max_value=_ALPHA_MAX
            )
            visibility = transmittance * fragment_alpha
            color = color + fragment_rgba[:3] * visibility[None]
            fragment_depth = (bary * triangle[:, 2, None, None]).sum(dim=0)
            accumulated_alpha = accumulated_alpha + visibility
            if depth_mode == "median":
                selected = ~median_selected & (accumulated_alpha >= 0.5)
                median_depth = th.where(selected, fragment_depth, median_depth)
                median_selected = median_selected | selected
            else:
                weighted_depth = weighted_depth + visibility * fragment_depth
            transmittance = transmittance * (1.0 - fragment_alpha)

        background_batch = _background_for_batch(background, batch_index, batch_size, v)
        color = color + background_batch[:, None, None] * transmittance[None]
        colors.append(color)
        alphas.append(accumulated_alpha[None])
        depths.append(
            _reference_depth(
                weighted_depth,
                accumulated_alpha,
                transmittance,
                median_depth,
                depth_mode,
                background_depth,
                depth_alpha_threshold,
            )[None]
        )

    return th.stack(colors, dim=0), th.stack(alphas, dim=0), th.stack(depths, dim=0)


def _expand_cuda_faces(vi: th.Tensor, batch_size: int) -> th.Tensor:
    if vi.ndim == 2:
        vi = vi[None]
    if vi.ndim != 3 or vi.shape[2] != 3 or vi.shape[0] not in (1, batch_size):
        raise ValueError(
            "transparent render: vi must have shape [F, 3], [1, F, 3], or "
            f"[N, F, 3], but got {tuple(vi.shape)}"
        )
    if vi.dtype != th.int32:
        raise TypeError(
            f"transparent render: vi must have int32 dtype, but got {vi.dtype}"
        )
    return vi.expand(batch_size, -1, -1).contiguous()


def _expand_cuda_vertex_attribute(
    value: th.Tensor,
    batch_size: int,
    vertex_count: int,
    channels: int,
    name: str,
) -> th.Tensor:
    unbatched_dims = 1 if channels == 1 else 2
    if value.ndim == unbatched_dims:
        value = value[None]
    expected_dims = unbatched_dims + 1
    expected_tail = (vertex_count,) if channels == 1 else (vertex_count, channels)
    if (
        value.ndim != expected_dims
        or value.shape[0] not in (1, batch_size)
        or tuple(value.shape[1:]) != expected_tail
    ):
        raise ValueError(
            f"transparent render: invalid {name} shape {tuple(value.shape)}; "
            f"expected batch 1 or {batch_size} and tail {expected_tail}"
        )
    return value.expand(batch_size, *expected_tail).float().contiguous()


def _expand_cuda_background(
    background: Optional[th.Tensor],
    batch_size: int,
    like: th.Tensor,
) -> th.Tensor:
    if background is None:
        return like.new_zeros((batch_size, 3), dtype=th.float32)
    if background.ndim == 1:
        background = background[None]
    if (
        background.ndim != 2
        or background.shape[0] not in (1, batch_size)
        or background.shape[1] != 3
    ):
        raise ValueError(
            "transparent render: background must have shape [3], [1, 3], or "
            f"[N, 3], but got {tuple(background.shape)}"
        )
    return background.expand(batch_size, -1).float().contiguous()


def _prepare_cuda_common(
    v: th.Tensor,
    vi: th.Tensor,
    uv: th.Tensor,
    opacity: Optional[th.Tensor],
    background: Optional[th.Tensor],
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    batch_size, vertex_count, _ = v.shape
    v_cuda = v.float().contiguous()
    vi_cuda = _expand_cuda_faces(vi, batch_size)
    uv_cuda = _expand_cuda_vertex_attribute(uv, batch_size, vertex_count, 2, "uv")
    opacity_cuda = (
        v_cuda.new_ones((batch_size, vertex_count))
        if opacity is None
        else _expand_cuda_vertex_attribute(
            opacity, batch_size, vertex_count, 1, "opacity"
        )
    )
    background_cuda = _expand_cuda_background(background, batch_size, v_cuda)
    return v_cuda, vi_cuda, uv_cuda, opacity_cuda, background_cuda


@th.compiler.disable
def render_transparent(
    v: th.Tensor,
    vi: th.Tensor,
    uv: th.Tensor,
    texture: th.Tensor,
    height: int,
    width: int,
    opacity: Optional[th.Tensor] = None,
    background: Optional[th.Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "border",
    transmittance_epsilon: float = 0.0,
    rgb_activation: str = "relu",
    alpha_activation: str = "relu_density",
    edge_gradients: bool = True,
    *,
    depth_mode: str = "expected",
    background_depth: float = 1000.0,
    depth_alpha_threshold: float = 1.0e-6,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Render depth-sorted transparent textured triangles.

    Triangles are composited front-to-back in stable mean-camera-depth order,
    with face index breaking ties. Interpolation is perspective-correct.
    Intersecting or depth-crossing triangles can composite in the wrong order;
    per-pixel sorting is not supported.

    Gradients propagate through texture sampling, interpolation, opacity, and
    compositing. Visibility-boundary gradients are enabled by default through
    ``edge_gradients=True``. Ordinary shared tessellation edges and zero-alpha
    borders contribute no boundary term. Set ``transmittance_epsilon`` only
    for inference: nonzero values permit early termination.

    This function requires CUDA. :func:`render_transparent_ref` is available
    for CPU use and supports smooth gradients only.

    Args:
        v: Screen-space vertices ``[N, V, 3]``. X/Y use DRTK pixel coordinates
            and smaller positive Z values are nearer.
        vi: Int32 triangle indices ``[F, 3]``, ``[1, F, 3]``, or
            ``[N, F, 3]``.
        uv: Per-vertex normalized texture coordinates in ``[0, 1]``.
        texture: RGB or RGBA texture ``[N or 1, 3 or 4, Ht, Wt]``.
        height: Output image height.
        width: Output image width.
        opacity: Optional per-vertex material opacity ``[N or 1, V]``.
        background: Optional RGB background ``[3]`` or ``[N or 1, 3]``.
        mode: Texture interpolation mode: ``"bilinear"``, ``"nearest"``, or
            ``"bicubic"``.
        padding_mode: Texture padding mode: ``"zeros"``, ``"border"``, or
            ``"reflection"``. Texture coordinates always use
            ``align_corners=False`` semantics.
        transmittance_epsilon: Optional inference-only early-termination
            threshold. It must be zero when gradients are enabled.
        rgb_activation: Activation applied to filtered RGB. The default
            ``"relu"`` computes ``max(x, 0)``. Alternatives are ``"identity"``,
            ``"shifted_sigmoid"`` (``sigmoid(x - 0.5)``), and ``"clamp"``.
        alpha_activation: Activation applied to filtered texture alpha. The
            default ``"relu_density"`` computes ``1 - exp(-max(x, 0))``.
            Alternatives are ``"shifted_sigmoid"``
            (``sigmoid(x - 0.5)``), Instant-NGP-style ``"exp_density"``
            (``1 - exp(-exp(x))``, with ``x`` capped at 15 before
            exponentiation), and ``"clamp"``.

        edge_gradients: Add transparent coverage-boundary gradients to vertex
            X/Y motion. Defaults to ``True``; set to ``False`` to disable.
            This is a filtered-surrogate estimator, not the pointwise derivative
            of hard coverage. It uses
            the same mean-triangle-depth order and adds no inference work.
            Indexed shared tessellation edges and zero-alpha edge traces are
            suppressed; folded shared boundaries are composited as a bundle.
            The reference functions retain smooth/interior gradients only.

        depth_mode: ``"expected"`` (default) composites camera-space Z with
            an opaque background at ``background_depth``. ``"accumulated"``
            returns only the foreground sum of ``transmittance * alpha * Z``.
            ``"foreground_mean"`` divides that sum by foreground alpha only
            above ``depth_alpha_threshold``; other pixels return background
            depth with zero smooth depth gradient. ``"median"`` selects the first
            fragment whose cumulative foreground alpha reaches 0.5, or the
            background if no fragment does. All modes use the same stable
            mean-triangle-depth order as RGB, without per-pixel sorting.
        background_depth: Positive finite background camera Z, in the same
            units as ``v[..., 2]``. Defaults to 1000. Set it to an appropriate
            far depth for the scene; it is independent of RGB ``background``.
            Infinity is not supported. A distant background increases the
            depth-loss signal for opacity and silhouette changes.
        depth_alpha_threshold: Minimum foreground alpha for a valid mean
            depth. Defaults to 1e-6. The division is not evaluated through
            small or zero alpha; invalid pixels have zero smooth depth gradient.
            This threshold only affects ``"foreground_mean"``.

    Returns:
        A tuple ``(rgb, alpha, depth)`` with shapes ``[N, 3, H, W]``,
        ``[N, 1, H, W]``, and ``[N, 1, H, W]``. RGB is premultiplied unless
        a background is supplied. Alpha always describes foreground opacity,
        even when depth includes an opaque background. Fragment depth is
        perspective-correct interpolated camera Z, not ray distance or the
        triangle-center depth used for sorting.

        Expected, accumulated, and valid foreground-mean depth differentiate
        geometry and opacity. Median differentiates the selected fragment's
        depth, with no smooth opacity gradient through its discrete selection.
        Ordering and the foreground-mean validity test are also discrete.
        Boundary gradients include the chosen depth output when
        ``edge_gradients=True``; reference implementations retain only smooth
        derivatives with coverage fixed.
    """

    depth_mode_enum = _validate_depth_options(
        depth_mode, background_depth, depth_alpha_threshold
    )
    _validate_common(v, vi, height, width)
    mode_enum, padding_mode_enum = _validate_sampling_options(mode, padding_mode)
    rgb_activation_enum, alpha_activation_enum = _validate_texture_activation_options(
        rgb_activation, alpha_activation
    )
    if not 0.0 <= transmittance_epsilon < 1.0:
        raise ValueError(
            "transparent render: transmittance_epsilon must be in [0, 1), but "
            f"got {transmittance_epsilon}"
        )
    v_cuda, vi_cuda, uv_cuda, opacity_cuda, background_cuda = _prepare_cuda_common(
        v, vi, uv, opacity, background
    )
    texture_cuda = texture.float().contiguous()
    outputs = _render_with_texture(
        v_cuda,
        vi_cuda,
        uv_cuda,
        texture_cuda,
        opacity_cuda,
        background_cuda,
        height,
        width,
        mode_enum,
        padding_mode_enum,
        transmittance_epsilon,
        rgb_activation_enum,
        alpha_activation_enum,
        edge_gradients,
        depth_mode_enum,
        background_depth,
        depth_alpha_threshold,
    )
    return outputs[0], outputs[1], outputs[2]


@th.compiler.disable
def render_transparent_mlp(
    v: th.Tensor,
    vi: th.Tensor,
    uv: th.Tensor,
    texture: th.Tensor,
    weights: List[th.Tensor],
    biases: List[th.Tensor],
    height: int,
    width: int,
    opacity: Optional[th.Tensor] = None,
    background: Optional[th.Tensor] = None,
    negative_slope: float = 0.2,
    mode: str = "bilinear",
    padding_mode: str = "border",
    transmittance_epsilon: float = 0.0,
    rgb_activation: str = "shifted_relu",
    alpha_activation: str = "shifted_relu_density",
    edge_gradients: bool = True,
    *,
    depth_mode: str = "expected",
    background_depth: float = 1000.0,
    depth_alpha_threshold: float = 1.0e-6,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Render transparent triangles from one neural feature texture.

    Samples ``texture`` at each fragment's interpolated ``uv`` and evaluates
    an appearance MLP to obtain RGBA. The MLP must have four or five affine
    layers, hidden width eight, and LeakyReLU hidden activations. Gradients
    propagate to geometry, UVs, texture, opacity, background, weights, and biases.

    ``rgb_activation`` defaults to ``"shifted_relu"``, which computes
    ``max(x + 0.5, 0)``. It also accepts ``"shifted_identity"`` (``x + 0.5``),
    ``"sigmoid"`` (``sigmoid(x)``), and ``"shifted_clamp"``
    (``clamp(x + 0.5, 0, 1)``). ``alpha_activation`` defaults to
    ``"shifted_relu_density"``, which computes
    ``1 - exp(-max(x + 0.5, 0))``. It also accepts ``"sigmoid"``,
    ``"exp_density"`` (``1 - exp(-exp(x))``, with ``x`` capped at 15 before
    exponentiation), and ``"shifted_clamp"``.
    The shifts make a zero-centered MLP output represent a useful mid-range
    value. Activation follows feature filtering and MLP evaluation.

    Triangles are composited in ascending mean camera-depth order, with face
    index breaking ties.
    This approximates per-pixel ordering; intersecting or depth-crossing
    triangles may composite in the wrong order. There is no hard z-buffer.
    Hard top-left coverage and clamped interpolated
    material opacity remain outside the MLP. The resulting fragment alpha is
    clamped to [0, 1] before front-to-back transmittance compositing. For soft
    boundaries, supply explicitly feathered geometry or opacity.

    This entry point requires CUDA. Use :func:`render_transparent_mlp_ref` for
    the pure-PyTorch correctness implementation.

    Args:
        v: Screen-space vertices ``[N, V, 3]``. X/Y use DRTK pixel coordinates
            and smaller positive Z values are nearer.
        vi: Int32 triangle indices ``[F, 3]``, ``[1, F, 3]``, or
            ``[N, F, 3]``.
        uv: Per-vertex normalized coordinates ``[V, 2]``, ``[1, V, 2]``, or
            ``[N, V, 2]`` for ``texture``.
        texture: Feature texture ``[N or 1, C, Ht, Wt]``, where
            ``1 <= C <= 32``.
        weights: Four or five affine weight tensors. Hidden outputs have width
            eight and the final weight has shape ``[4, 8]``.
        biases: One bias tensor per weight; the final bias has shape ``[4]``.
        height: Output image height.
        width: Output image width.
        opacity: Optional per-vertex material opacity ``[V]``, ``[1, V]``, or
            ``[N, V]``. The default is one.
        background: Optional RGB background ``[3]`` or ``[N or 1, 3]``.
        negative_slope: LeakyReLU slope for hidden MLP activations.
        mode: Texture interpolation mode: ``"bilinear"``, ``"nearest"``, or
            ``"bicubic"``.
        padding_mode: Texture padding mode: ``"zeros"``, ``"border"``, or
            ``"reflection"``. Sampling always uses ``align_corners=False``
            semantics.
        transmittance_epsilon: Optional inference-only early-termination
            threshold in ``[0, 1)``. It must be zero when gradients are
            enabled.
        rgb_activation: Final RGB activation described above.
        alpha_activation: Final appearance-alpha activation described above.

        edge_gradients: Add transparent coverage-boundary gradients to vertex
            X/Y motion. Defaults to ``True``; set to ``False`` to disable.
            This is a filtered-surrogate estimator, not the pointwise derivative
            of hard coverage. It uses
            the same mean-triangle-depth order and adds no inference work.
            Indexed shared tessellation edges and zero-alpha edge traces are
            suppressed; folded shared boundaries are composited as a bundle.
            The reference functions retain smooth/interior gradients only.

        depth_mode: ``"expected"`` (default) composites camera-space Z with
            an opaque background at ``background_depth``. ``"accumulated"``
            returns only the foreground sum of ``transmittance * alpha * Z``.
            ``"foreground_mean"`` divides that sum by foreground alpha only
            above ``depth_alpha_threshold``; other pixels return background
            depth with zero smooth depth gradient. ``"median"`` selects the first
            fragment whose cumulative foreground alpha reaches 0.5, or the
            background if no fragment does. All modes use the same stable
            mean-triangle-depth order as RGB, without per-pixel sorting.
        background_depth: Positive finite background camera Z, in the same
            units as ``v[..., 2]``. Defaults to 1000. Set it to an appropriate
            far depth for the scene; it is independent of RGB ``background``.
            Infinity is not supported. A distant background increases the
            depth-loss signal for opacity and silhouette changes.
        depth_alpha_threshold: Minimum foreground alpha for a valid mean
            depth. Defaults to 1e-6. The division is not evaluated through
            small or zero alpha; invalid pixels have zero smooth depth gradient.
            This threshold only affects ``"foreground_mean"``.

    Returns:
        A tuple ``(rgb, alpha, depth)`` with shapes ``[N, 3, H, W]``,
        ``[N, 1, H, W]``, and ``[N, 1, H, W]``. RGB is premultiplied unless
        a background is supplied. Alpha always describes foreground opacity,
        even when depth includes an opaque background. Fragment depth is
        perspective-correct interpolated camera Z, not ray distance or the
        triangle-center depth used for sorting.

        Expected, accumulated, and valid foreground-mean depth differentiate
        geometry and opacity. Median differentiates the selected fragment's
        depth, with no smooth opacity gradient through its discrete selection.
        Ordering and the foreground-mean validity test are also discrete.
        Boundary gradients include the chosen depth output when
        ``edge_gradients=True``; reference implementations retain only smooth
        derivatives with coverage fixed.
    """

    empty_texture = texture.new_empty((1, 0, 1, 1))
    return _render_transparent_mlp_impl(
        v,
        vi,
        uv,
        uv,
        texture,
        empty_texture,
        weights,
        biases,
        height,
        width,
        opacity,
        background,
        negative_slope,
        mode,
        padding_mode,
        transmittance_epsilon,
        rgb_activation,
        alpha_activation,
        edge_gradients,
        depth_mode=depth_mode,
        background_depth=background_depth,
        depth_alpha_threshold=depth_alpha_threshold,
    )


@th.compiler.disable
def render_transparent_mlp_two_textures(
    v: th.Tensor,
    vi: th.Tensor,
    uv0: th.Tensor,
    uv1: th.Tensor,
    texture0: th.Tensor,
    texture1: th.Tensor,
    weights: List[th.Tensor],
    biases: List[th.Tensor],
    height: int,
    width: int,
    opacity: Optional[th.Tensor] = None,
    background: Optional[th.Tensor] = None,
    negative_slope: float = 0.2,
    mode: str = "bilinear",
    padding_mode: str = "border",
    transmittance_epsilon: float = 0.0,
    rgb_activation: str = "shifted_relu",
    alpha_activation: str = "shifted_relu_density",
    edge_gradients: bool = True,
    *,
    depth_mode: str = "expected",
    background_depth: float = 1000.0,
    depth_alpha_threshold: float = 1.0e-6,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Render transparent triangles from two neural feature textures.

    Samples ``texture0`` at interpolated ``uv0`` and ``texture1`` at ``uv1``,
    concatenates their feature vectors in that order, and evaluates an
    appearance MLP to obtain RGBA. The MLP must have four or five affine
    layers, hidden width eight, and LeakyReLU hidden activations. Gradients
    propagate to geometry, both UV sets and textures, opacity, background,
    weights, and biases.

    ``rgb_activation`` defaults to ``"shifted_relu"``, which computes
    ``max(x + 0.5, 0)``. It also accepts ``"shifted_identity"`` (``x + 0.5``),
    ``"sigmoid"`` (``sigmoid(x)``), and ``"shifted_clamp"``
    (``clamp(x + 0.5, 0, 1)``). ``alpha_activation`` defaults to
    ``"shifted_relu_density"``, which computes
    ``1 - exp(-max(x + 0.5, 0))``. It also accepts ``"sigmoid"``,
    ``"exp_density"`` (``1 - exp(-exp(x))``, with ``x`` capped at 15 before
    exponentiation), and ``"shifted_clamp"``.
    The shifts make a zero-centered MLP output represent a useful mid-range
    value. Activation follows feature filtering and MLP evaluation.

    Triangles are composited in ascending mean camera-depth order, with face
    index breaking ties.
    This approximates per-pixel ordering; intersecting or depth-crossing
    triangles may composite in the wrong order. There is no hard z-buffer.
    Hard top-left coverage and clamped interpolated
    material opacity remain outside the MLP. The resulting fragment alpha is
    clamped to [0, 1] before front-to-back transmittance compositing. For soft
    boundaries, supply explicitly feathered geometry or opacity.

    This entry point requires CUDA. Use
    :func:`render_transparent_mlp_two_textures_ref` for the pure-PyTorch
    correctness implementation.

    The caller must gate features from occluded or unrelated surfaces and
    provide visibility/provenance validity information where needed. Separate
    UV coordinates do not perform this gating automatically.

    Args:
        v: Screen-space vertices ``[N, V, 3]``. X/Y use DRTK pixel coordinates
            and smaller positive Z values are nearer.
        vi: Int32 triangle indices ``[F, 3]``, ``[1, F, 3]``, or
            ``[N, F, 3]``.
        uv0: Per-vertex normalized coordinates ``[V, 2]``, ``[1, V, 2]``, or
            ``[N, V, 2]`` for ``texture0``.
        uv1: Per-vertex normalized coordinates with the same supported shapes
            as ``uv0`` for ``texture1``.
        texture0: First feature texture ``[N or 1, C0, H0, W0]``.
        texture1: Second feature texture ``[N or 1, C1, H1, W1]``. The channel
            counts must satisfy ``1 <= C0 + C1 <= 32``.
        weights: Four or five affine weight tensors. Hidden outputs have width
            eight and the final weight has shape ``[4, 8]``.
        biases: One bias tensor per weight; the final bias has shape ``[4]``.
        height: Output image height.
        width: Output image width.
        opacity: Optional per-vertex material opacity ``[V]``, ``[1, V]``, or
            ``[N, V]``. The default is one.
        background: Optional RGB background ``[3]`` or ``[N or 1, 3]``.
        negative_slope: LeakyReLU slope for hidden MLP activations.
        mode: Interpolation mode shared by both textures: ``"bilinear"``,
            ``"nearest"``, or ``"bicubic"``.
        padding_mode: Padding mode shared by both textures: ``"zeros"``,
            ``"border"``, or ``"reflection"``. Sampling always uses
            ``align_corners=False`` semantics.
        transmittance_epsilon: Optional inference-only early-termination
            threshold in ``[0, 1)``. It must be zero when gradients are
            enabled.
        rgb_activation: Final RGB activation described above.
        alpha_activation: Final appearance-alpha activation described above.

        edge_gradients: Add transparent coverage-boundary gradients to vertex
            X/Y motion. Defaults to ``True``; set to ``False`` to disable.
            This is a filtered-surrogate estimator, not the pointwise derivative
            of hard coverage. It uses
            the same mean-triangle-depth order and adds no inference work.
            Indexed shared tessellation edges and zero-alpha edge traces are
            suppressed; folded shared boundaries are composited as a bundle.
            The reference functions retain smooth/interior gradients only.

        depth_mode: ``"expected"`` (default) composites camera-space Z with
            an opaque background at ``background_depth``. ``"accumulated"``
            returns only the foreground sum of ``transmittance * alpha * Z``.
            ``"foreground_mean"`` divides that sum by foreground alpha only
            above ``depth_alpha_threshold``; other pixels return background
            depth with zero smooth depth gradient. ``"median"`` selects the first
            fragment whose cumulative foreground alpha reaches 0.5, or the
            background if no fragment does. All modes use the same stable
            mean-triangle-depth order as RGB, without per-pixel sorting.
        background_depth: Positive finite background camera Z, in the same
            units as ``v[..., 2]``. Defaults to 1000. Set it to an appropriate
            far depth for the scene; it is independent of RGB ``background``.
            Infinity is not supported. A distant background increases the
            depth-loss signal for opacity and silhouette changes.
        depth_alpha_threshold: Minimum foreground alpha for a valid mean
            depth. Defaults to 1e-6. The division is not evaluated through
            small or zero alpha; invalid pixels have zero smooth depth gradient.
            This threshold only affects ``"foreground_mean"``.

    Returns:
        A tuple ``(rgb, alpha, depth)`` with shapes ``[N, 3, H, W]``,
        ``[N, 1, H, W]``, and ``[N, 1, H, W]``. RGB is premultiplied unless
        a background is supplied. Alpha always describes foreground opacity,
        even when depth includes an opaque background. Fragment depth is
        perspective-correct interpolated camera Z, not ray distance or the
        triangle-center depth used for sorting.

        Expected, accumulated, and valid foreground-mean depth differentiate
        geometry and opacity. Median differentiates the selected fragment's
        depth, with no smooth opacity gradient through its discrete selection.
        Ordering and the foreground-mean validity test are also discrete.
        Boundary gradients include the chosen depth output when
        ``edge_gradients=True``; reference implementations retain only smooth
        derivatives with coverage fixed.
    """

    return _render_transparent_mlp_impl(
        v,
        vi,
        uv0,
        uv1,
        texture0,
        texture1,
        weights,
        biases,
        height,
        width,
        opacity,
        background,
        negative_slope,
        mode,
        padding_mode,
        transmittance_epsilon,
        rgb_activation,
        alpha_activation,
        edge_gradients,
        depth_mode=depth_mode,
        background_depth=background_depth,
        depth_alpha_threshold=depth_alpha_threshold,
    )


@th.compiler.disable
def _render_transparent_mlp_impl(
    v: th.Tensor,
    vi: th.Tensor,
    uv0: th.Tensor,
    uv1: th.Tensor,
    texture0: th.Tensor,
    texture1: th.Tensor,
    weights: List[th.Tensor],
    biases: List[th.Tensor],
    height: int,
    width: int,
    opacity: Optional[th.Tensor] = None,
    background: Optional[th.Tensor] = None,
    negative_slope: float = 0.2,
    mode: str = "bilinear",
    padding_mode: str = "border",
    transmittance_epsilon: float = 0.0,
    rgb_activation: str = "shifted_relu",
    alpha_activation: str = "shifted_relu_density",
    edge_gradients: bool = True,
    *,
    depth_mode: str = "expected",
    background_depth: float = 1000.0,
    depth_alpha_threshold: float = 1.0e-6,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Shared implementation for the one- and two-texture CUDA APIs."""

    depth_mode_enum = _validate_depth_options(
        depth_mode, background_depth, depth_alpha_threshold
    )
    _validate_common(v, vi, height, width)
    _validate_mlp(texture0.shape[1] + texture1.shape[1], weights, biases)
    mode_enum, padding_mode_enum = _validate_sampling_options(mode, padding_mode)
    rgb_activation_enum, alpha_activation_enum = _validate_mlp_activation_options(
        rgb_activation, alpha_activation
    )
    if not 0.0 <= transmittance_epsilon < 1.0:
        raise ValueError(
            "transparent render MLP: transmittance_epsilon must be in [0, 1)"
        )
    v_cuda, vi_cuda, uv_cuda, opacity_cuda, background_cuda = _prepare_cuda_common(
        v, vi, uv0, opacity, background
    )
    uv1_cuda = _expand_cuda_vertex_attribute(uv1, v.shape[0], v.shape[1], 2, "uv1")
    texture0_cuda = texture0.float().contiguous()
    texture1_cuda = texture1.float().contiguous()
    weights_cuda = [weight.float().contiguous() for weight in weights]
    biases_cuda = [bias.float().contiguous() for bias in biases]
    outputs = _render_with_mlp(
        v_cuda,
        vi_cuda,
        uv_cuda,
        uv1_cuda,
        texture0_cuda,
        texture1_cuda,
        weights_cuda,
        biases_cuda,
        opacity_cuda,
        background_cuda,
        height,
        width,
        negative_slope,
        mode_enum,
        padding_mode_enum,
        transmittance_epsilon,
        rgb_activation_enum,
        alpha_activation_enum,
        edge_gradients,
        depth_mode_enum,
        background_depth,
        depth_alpha_threshold,
    )
    return outputs[0], outputs[1], outputs[2]

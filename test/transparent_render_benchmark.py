# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Microbenchmark for DRTK's fused transparent triangle renderer.

Run this after building DRTK in place::

    python test/transparent_render_benchmark.py --resolution 128 --layers 4

The forward measurements include triangle binning and sorting. Backward
measurements use the public operators' native C++ autograd nodes and reuse the
saved sorted tile lists. Transparent boundary gradients are enabled by
default; pass --no-edge-gradients to measure smooth backward alone. The
benchmark does not depend on private kernel entry points.
"""

import argparse
import importlib
import json
from typing import Any, Callable, Dict, List, Tuple

import torch as th


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--layers", type=int, choices=(4, 5), default=4)
    parser.add_argument("--feature-channels-per-texture", type=int, default=4)
    parser.add_argument(
        "--edge-gradients", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def _make_pixel_grid_mesh(
    resolution: int,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    y, x = th.meshgrid(
        th.arange(resolution, device="cuda", dtype=th.float32),
        th.arange(resolution, device="cuda", dtype=th.float32),
        indexing="ij",
    )
    depth = th.ones_like(x)
    vertices = th.stack((x, y, depth), dim=-1).reshape(1, -1, 3).contiguous()
    uv = (
        th.stack(((x + 0.5) / resolution, (y + 0.5) / resolution), dim=-1)
        .reshape(1, -1, 2)
        .contiguous()
    )

    vertex_ids = th.arange(
        resolution * resolution, device="cuda", dtype=th.int32
    ).reshape(resolution, resolution)
    top_left = vertex_ids[:-1, :-1].reshape(-1)
    top_right = vertex_ids[:-1, 1:].reshape(-1)
    bottom_left = vertex_ids[1:, :-1].reshape(-1)
    bottom_right = vertex_ids[1:, 1:].reshape(-1)
    faces = (
        th.stack(
            (
                th.stack((top_left, top_right, bottom_left), dim=1),
                th.stack((top_right, bottom_right, bottom_left), dim=1),
            ),
            dim=1,
        )
        .reshape(1, -1, 3)
        .contiguous()
    )
    return vertices, faces, uv


def _make_mlp(
    input_channels: int,
    layer_count: int,
) -> Tuple[List[th.Tensor], List[th.Tensor]]:
    weights: List[th.Tensor] = []
    biases: List[th.Tensor] = []
    current_channels = input_channels
    for layer in range(layer_count):
        output_channels = 4 if layer == layer_count - 1 else 8
        weights.append(
            th.randn(
                (output_channels, current_channels),
                dtype=th.float32,
                device="cuda",
            )
            * 0.1
        )
        biases.append(th.zeros((output_channels,), dtype=th.float32, device="cuda"))
        current_channels = output_channels
    return weights, biases


def _time_cuda(
    function: Callable[[], Any],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    th.cuda.synchronize()
    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _candidate_tests_per_pixel(
    vertices: th.Tensor,
    faces: th.Tensor,
    resolution: int,
    tile_size: int = 4,
) -> float:
    """Reproduce the renderer's hard-bounds tile count for diagnostics."""

    triangle_xy = vertices[0, faces[0].long(), :2]
    minimum = triangle_xy.amin(dim=1)
    maximum = triangle_xy.amax(dim=1)
    tile_count = (resolution + tile_size - 1) // tile_size
    minimum_tile = th.floor(minimum / tile_size).to(th.int64).clamp(0, tile_count)
    maximum_tile = (th.floor(maximum / tile_size).to(th.int64) + 1).clamp(0, tile_count)
    valid = (
        (maximum[:, 0] >= 0)
        & (maximum[:, 1] >= 0)
        & (minimum[:, 0] <= resolution - 1)
        & (minimum[:, 1] <= resolution - 1)
    )
    minimum_pixel = minimum_tile * tile_size
    maximum_pixel = (maximum_tile * tile_size).clamp(max=resolution)
    pixels_per_bounds = (maximum_pixel - minimum_pixel).prod(dim=1)
    total_tests = th.where(valid, pixels_per_bounds, 0).sum().item()
    return total_tests / (resolution * resolution)


def main() -> None:
    args = _parse_args()
    if not th.cuda.is_available():
        raise RuntimeError("The transparent renderer benchmark requires CUDA")
    if args.resolution < 2:
        raise ValueError("--resolution must be at least two")
    if not 1 <= args.feature_channels_per_texture <= 16:
        raise ValueError(
            "--feature-channels-per-texture must be in [1, 16] so the two "
            "textures fit the renderer's 32-channel limit"
        )

    th.manual_seed(20261103)
    transparent_render = importlib.import_module("drtk.transparent_render")
    vertices, faces, uv = _make_pixel_grid_mesh(args.resolution)
    vertex_count = vertices.shape[1]
    opacity = th.full((1, vertex_count), 0.65, device="cuda")
    background = th.zeros((1, 3), device="cuda")
    rgba = th.rand((1, 4, args.resolution, args.resolution), device="cuda")
    rgba[:, 3].mul_(0.5).add_(0.25)

    feature_channels = args.feature_channels_per_texture
    texture = th.rand(
        (1, feature_channels, args.resolution, args.resolution), device="cuda"
    )
    reference_texture = th.rand_like(texture)
    weights, biases = _make_mlp(feature_channels * 2, args.layers)

    def render_with_texture_forward() -> Any:
        return transparent_render.render_transparent(
            vertices,
            faces,
            uv,
            rgba,
            args.resolution,
            args.resolution,
            opacity=opacity,
            background=background,
        )

    def render_with_mlp_forward() -> Any:
        return transparent_render.render_transparent_mlp_two_textures(
            vertices,
            faces,
            uv,
            uv,
            texture,
            reference_texture,
            weights,
            biases,
            args.resolution,
            args.resolution,
            opacity=opacity,
            background=background,
        )

    texture_grad_inputs = [
        value.detach().clone().requires_grad_()
        for value in (vertices, uv, rgba, opacity, background)
    ]
    texture_grad_output = transparent_render.render_transparent(
        texture_grad_inputs[0],
        faces,
        texture_grad_inputs[1],
        texture_grad_inputs[2],
        args.resolution,
        args.resolution,
        opacity=texture_grad_inputs[3],
        edge_gradients=args.edge_gradients,
        background=texture_grad_inputs[4],
    )
    texture_grad_outputs = (
        th.randn_like(texture_grad_output[0]),
        th.randn_like(texture_grad_output[1]),
    )

    def render_with_texture_backward() -> Any:
        return th.autograd.grad(
            texture_grad_output,
            texture_grad_inputs,
            texture_grad_outputs,
            retain_graph=True,
        )

    mlp_base_grad_inputs = [
        value.detach().clone().requires_grad_()
        for value in (
            vertices,
            uv,
            uv,
            texture,
            reference_texture,
            opacity,
            background,
        )
    ]
    weight_grad_inputs = [
        weight.detach().clone().requires_grad_() for weight in weights
    ]
    bias_grad_inputs = [bias.detach().clone().requires_grad_() for bias in biases]
    mlp_grad_inputs = [
        *mlp_base_grad_inputs,
        *weight_grad_inputs,
        *bias_grad_inputs,
    ]
    mlp_grad_output = transparent_render.render_transparent_mlp_two_textures(
        mlp_base_grad_inputs[0],
        faces,
        mlp_base_grad_inputs[1],
        mlp_base_grad_inputs[2],
        mlp_base_grad_inputs[3],
        mlp_base_grad_inputs[4],
        weight_grad_inputs,
        bias_grad_inputs,
        args.resolution,
        args.resolution,
        opacity=mlp_base_grad_inputs[5],
        edge_gradients=args.edge_gradients,
        background=mlp_base_grad_inputs[6],
    )
    mlp_grad_outputs = (
        th.randn_like(mlp_grad_output[0]),
        th.randn_like(mlp_grad_output[1]),
    )

    def render_with_mlp_backward() -> Any:
        return th.autograd.grad(
            mlp_grad_output,
            mlp_grad_inputs,
            mlp_grad_outputs,
            retain_graph=True,
        )

    total_feature_channels = feature_channels * 2
    mlp_macs = 8 * total_feature_channels + (args.layers - 2) * 8 * 8 + 4 * 8
    result: Dict[str, Any] = {
        "gpu": th.cuda.get_device_name(),
        "edge_gradients": args.edge_gradients,
        "torch": th.__version__,
        "resolution": args.resolution,
        "vertices": vertex_count,
        "triangles": faces.shape[1],
        "tile_size": 4,
        "candidate_tests_per_pixel": _candidate_tests_per_pixel(
            vertices, faces, args.resolution
        ),
        "mlp_layers": args.layers,
        "mlp_width": 8,
        "total_feature_channels": total_feature_channels,
        "mlp_macs_per_accepted_fragment": mlp_macs,
        "render_with_texture_forward_ms": _time_cuda(
            render_with_texture_forward, args.warmup, args.iterations
        ),
        "render_with_mlp_forward_ms": _time_cuda(
            render_with_mlp_forward, args.warmup, args.iterations
        ),
        "render_with_texture_backward_ms": _time_cuda(
            render_with_texture_backward, args.warmup, args.iterations
        ),
        "render_with_mlp_backward_ms": _time_cuda(
            render_with_mlp_backward, args.warmup, args.iterations
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

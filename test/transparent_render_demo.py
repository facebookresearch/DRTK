# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Optimize two transparent meshes over a fixed opaque mesh.

The renderer produces premultiplied color and accumulated alpha for three
depth-sorted 4 x 4 patches: an opaque rear layer and two semitransparent layers
above it. The transparent boundaries are fixed at zero opacity, while their
eight interior opacities and eight interior vertex positions are optimized. A
checkerboard is added only when saving visualization images; it is not rendered
geometry and does not participate in the loss.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch as th
from drtk import render_transparent
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image


_GRID_SIZE = 4
_VERTICES_PER_LAYER = _GRID_SIZE * _GRID_SIZE
_LOCAL_INTERIOR_VERTEX_IDS = (5, 6, 9, 10)
_MIDDLE_VERTEX_OFFSET = _VERTICES_PER_LAYER
_TOP_VERTEX_OFFSET = 2 * _VERTICES_PER_LAYER
_MIDDLE_INTERIOR_VERTEX_IDS = tuple(
    _MIDDLE_VERTEX_OFFSET + vertex_id for vertex_id in _LOCAL_INTERIOR_VERTEX_IDS
)
_TOP_INTERIOR_VERTEX_IDS = tuple(
    _TOP_VERTEX_OFFSET + vertex_id for vertex_id in _LOCAL_INTERIOR_VERTEX_IDS
)
_TRANSPARENT_INTERIOR_VERTEX_IDS = (
    *_MIDDLE_INTERIOR_VERTEX_IDS,
    *_TOP_INTERIOR_VERTEX_IDS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("transparent_render_demo_output")
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--geometry-learning-rate", type=float, default=0.2)
    parser.add_argument("--opacity-learning-rate", type=float, default=0.05)
    return parser.parse_args()


def _make_texture(device: th.device, tile_size: int = 256) -> th.Tensor:
    """Create one simple opaque appearance tile for each mesh layer."""

    axis = (th.arange(tile_size, device=device, dtype=th.float32) + 0.5) / tile_size
    v, u = th.meshgrid(axis, axis, indexing="ij")

    back_rgb = th.stack(
        (
            0.88 + 0.08 * u,
            0.25 + 0.12 * v,
            0.06 + 0.08 * (1.0 - u),
        ),
        dim=0,
    )
    back = th.cat((back_rgb, th.ones((1, tile_size, tile_size), device=device)))

    radius = th.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
    circle = ((0.42 - radius) / 0.035).clamp(0.0, 1.0)[None]
    middle_base = th.stack(
        (
            0.04 + 0.10 * v,
            0.48 + 0.20 * u,
            0.82 + 0.14 * (1.0 - v),
        ),
        dim=0,
    )
    circle_rgb = th.stack(
        (
            0.18 + 0.22 * u,
            0.72 + 0.18 * (1.0 - v),
            0.92 - 0.18 * u,
        ),
        dim=0,
    )
    middle_rgb = middle_base * (1.0 - circle) + circle_rgb * circle
    ring = th.exp(-0.5 * ((radius - 0.27) / 0.018) ** 2)[None]
    middle_rgb = middle_rgb * (1.0 - 0.75 * ring) + 0.98 * 0.75 * ring
    middle = th.cat((middle_rgb, th.ones((1, tile_size, tile_size), device=device)))

    diamond_distance = th.abs(u - 0.5) + 0.75 * th.abs(v - 0.5)
    diamond = ((0.36 - diamond_distance) / 0.035).clamp(0.0, 1.0)[None]
    top_base = th.stack(
        (
            0.48 + 0.20 * u,
            0.08 + 0.10 * (1.0 - v),
            0.58 + 0.18 * v,
        ),
        dim=0,
    )
    diamond_rgb = th.stack(
        (
            0.96 - 0.10 * v,
            0.68 + 0.18 * u,
            0.10 + 0.10 * (1.0 - u),
        ),
        dim=0,
    )
    top_rgb = top_base * (1.0 - diamond) + diamond_rgb * diamond
    top = th.cat((top_rgb, th.ones((1, tile_size, tile_size), device=device)))

    return th.cat((back, middle, top), dim=2)[None]


def _make_checkerboard(height: int, width: int, device: th.device) -> th.Tensor:
    cell_size = max(8, min(height, width) // 8)
    y = th.arange(height, device=device) // cell_size
    x = th.arange(width, device=device) // cell_size
    mask = ((y[:, None] + x[None, :]) % 2).to(dtype=th.bool)
    dark = th.tensor([0.22, 0.24, 0.28], device=device)[:, None, None]
    light = th.tensor([0.72, 0.74, 0.78], device=device)[:, None, None]
    return th.where(mask[None], light, dark)[None]


def _grid_faces(vertex_offset: int) -> List[List[int]]:
    faces: List[List[int]] = []
    for row in range(_GRID_SIZE - 1):
        for column in range(_GRID_SIZE - 1):
            top_left = vertex_offset + row * _GRID_SIZE + column
            top_right = top_left + 1
            bottom_left = top_left + _GRID_SIZE
            bottom_right = bottom_left + 1
            faces.extend(
                (
                    [top_left, top_right, bottom_left],
                    [top_right, bottom_right, bottom_left],
                )
            )
    return faces


def _make_scene(
    device: th.device,
    height: int,
    width: int,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    back_xy = th.tensor(
        [
            [24.0, 62.0],
            [78.0, 48.0],
            [132.0, 52.0],
            [186.0, 70.0],
            [18.0, 112.0],
            [74.0, 104.0],
            [130.0, 110.0],
            [190.0, 122.0],
            [26.0, 166.0],
            [80.0, 158.0],
            [138.0, 166.0],
            [196.0, 176.0],
            [38.0, 220.0],
            [92.0, 210.0],
            [148.0, 216.0],
            [202.0, 226.0],
        ],
        dtype=th.float32,
        device=device,
    )
    middle_xy = th.tensor(
        [
            [68.0, 28.0],
            [122.0, 24.0],
            [180.0, 32.0],
            [232.0, 50.0],
            [58.0, 82.0],
            [118.0, 88.0],
            [178.0, 82.0],
            [238.0, 98.0],
            [62.0, 142.0],
            [112.0, 146.0],
            [184.0, 152.0],
            [232.0, 164.0],
            [74.0, 204.0],
            [126.0, 216.0],
            [186.0, 210.0],
            [226.0, 218.0],
        ],
        dtype=th.float32,
        device=device,
    )
    top_xy = th.tensor(
        [
            [112.0, 18.0],
            [154.0, 22.0],
            [202.0, 34.0],
            [240.0, 58.0],
            [94.0, 60.0],
            [146.0, 66.0],
            [200.0, 76.0],
            [244.0, 100.0],
            [88.0, 108.0],
            [142.0, 118.0],
            [202.0, 126.0],
            [238.0, 150.0],
            [96.0, 158.0],
            [150.0, 170.0],
            [202.0, 180.0],
            [226.0, 204.0],
        ],
        dtype=th.float32,
        device=device,
    )
    back_vertices = th.cat(
        (back_xy, th.full((_VERTICES_PER_LAYER, 1), 1.6, device=device)), dim=1
    )
    middle_vertices = th.cat(
        (middle_xy, th.full((_VERTICES_PER_LAYER, 1), 1.2, device=device)), dim=1
    )
    top_vertices = th.cat(
        (top_xy, th.full((_VERTICES_PER_LAYER, 1), 0.8, device=device)), dim=1
    )
    target_vertices = th.cat((back_vertices, middle_vertices, top_vertices), dim=0)[
        None
    ]
    initial_vertices = target_vertices.clone()
    initial_vertices[:, list(_TRANSPARENT_INTERIOR_VERTEX_IDS), :2] = th.tensor(
        [
            [138.0, 68.0],
            [158.0, 110.0],
            [134.0, 174.0],
            [160.0, 130.0],
            [162.0, 52.0],
            [182.0, 104.0],
            [174.0, 142.0],
            [170.0, 100.0],
        ],
        dtype=th.float32,
        device=device,
    )

    image_scale = th.tensor(
        [width / 256.0, height / 256.0, 1.0],
        dtype=th.float32,
        device=device,
    )
    target_vertices = target_vertices * image_scale
    initial_vertices = initial_vertices * image_scale

    faces = th.tensor(
        _grid_faces(0)
        + _grid_faces(_MIDDLE_VERTEX_OFFSET)
        + _grid_faces(_TOP_VERTEX_OFFSET),
        dtype=th.int32,
        device=device,
    )
    back_uv = [
        [0.005 + 0.323 * column / 3.0, 0.01 + 0.98 * row / 3.0]
        for row in range(_GRID_SIZE)
        for column in range(_GRID_SIZE)
    ]
    middle_uv = [
        [0.338 + 0.323 * column / 3.0, 0.01 + 0.98 * row / 3.0]
        for row in range(_GRID_SIZE)
        for column in range(_GRID_SIZE)
    ]
    top_uv = [
        [0.671 + 0.324 * column / 3.0, 0.01 + 0.98 * row / 3.0]
        for row in range(_GRID_SIZE)
        for column in range(_GRID_SIZE)
    ]
    uv = th.tensor([*back_uv, *middle_uv, *top_uv], dtype=th.float32, device=device)[
        None
    ]
    return target_vertices, initial_vertices, faces, uv


def _render(
    vertices: th.Tensor,
    faces: th.Tensor,
    uv: th.Tensor,
    texture: th.Tensor,
    opacity: th.Tensor,
    height: int,
    width: int,
) -> Tuple[th.Tensor, th.Tensor]:
    return render_transparent(
        vertices,
        faces,
        uv,
        texture,
        height=height,
        width=width,
        opacity=opacity,
        rgb_activation="identity",
        alpha_activation="clamp",
    )


def _tensor_to_image(color: th.Tensor) -> Image.Image:
    pixels = (
        color[0]
        .detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=th.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(pixels)


def _draw_mesh(
    image: Image.Image,
    vertices: th.Tensor,
    faces: th.Tensor,
    line_color: Tuple[int, int, int],
    point_color: Tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    xy = vertices[0, :, :2].detach().cpu().tolist()
    face_indices = faces.detach().cpu().tolist()
    edges = {
        tuple(sorted((face[corner], face[(corner + 1) % 3])))
        for face in face_indices
        for corner in range(3)
    }
    scale = max(1, min(image.size) // 256)
    for first, second in sorted(edges):
        segment = (tuple(xy[first]), tuple(xy[second]))
        draw.line(segment, fill=(16, 20, 28), width=3 * scale)
        draw.line(segment, fill=line_color, width=scale)
    radius = 3 * scale
    for vertex_id in sorted({vertex for face in face_indices for vertex in face}):
        x, y = xy[vertex_id]
        draw.ellipse(
            (
                x - radius - scale,
                y - radius - scale,
                x + radius + scale,
                y + radius + scale,
            ),
            fill=(16, 20, 28),
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=point_color,
        )


def _draw_panel_label(panel: Image.Image, label: str, header_height: int) -> None:
    font = ImageFont.load_default()
    bounds = font.getbbox(label)
    label_size = (bounds[2] - bounds[0] + 2, bounds[3] - bounds[1] + 2)
    label_image = Image.new("RGBA", label_size, (0, 0, 0, 0))
    ImageDraw.Draw(label_image).text(
        (1 - bounds[0], 1 - bounds[1]),
        label,
        fill=(245, 248, 255, 255),
        font=font,
    )
    scale = max(1, (header_height * 2 // 3) // label_image.height)
    label_image = label_image.resize(
        (label_image.width * scale, label_image.height * scale),
        Image.Resampling.NEAREST,
    )
    panel.paste(
        label_image,
        (header_height // 3, (header_height - label_image.height) // 2),
        label_image,
    )


def _make_panel(
    image_tensor: th.Tensor,
    label: str,
    vertices: Optional[th.Tensor] = None,
    transparent_faces: Optional[Tuple[th.Tensor, th.Tensor]] = None,
) -> Image.Image:
    image = _tensor_to_image(image_tensor)
    if vertices is not None and transparent_faces is not None:
        _draw_mesh(
            image,
            vertices,
            transparent_faces[0],
            (245, 248, 255),
            (255, 196, 64),
        )
        _draw_mesh(
            image,
            vertices,
            transparent_faces[1],
            (255, 214, 242),
            (255, 92, 196),
        )
    header_height = max(48, image.height // 16)
    panel = Image.new("RGB", (image.width, image.height + header_height), (12, 16, 24))
    panel.paste(image, (0, header_height))
    _draw_panel_label(panel, label, header_height)
    return panel


def _join_panels(panels: Tuple[Image.Image, ...]) -> Image.Image:
    gap = max(8, panels[0].width // 128)
    comparison = Image.new(
        "RGB",
        (
            sum(panel.width for panel in panels) + gap * (len(panels) - 1),
            panels[0].height,
        ),
        (12, 16, 24),
    )
    x = 0
    for panel in panels:
        comparison.paste(panel, (x, 0))
        x += panel.width + gap
    return comparison


def _composite_over_checkerboard(
    rendered: Tuple[th.Tensor, th.Tensor], checkerboard: th.Tensor
) -> th.Tensor:
    return rendered[0] + (1.0 - rendered[1]) * checkerboard


def _write_outputs(
    output_dir: Path,
    texture: th.Tensor,
    checkerboard: th.Tensor,
    initial: Tuple[th.Tensor, th.Tensor],
    target: Tuple[th.Tensor, th.Tensor],
    optimized: Tuple[th.Tensor, th.Tensor],
    initial_vertices: th.Tensor,
    target_vertices: th.Tensor,
    optimized_vertices: th.Tensor,
    transparent_faces: Tuple[th.Tensor, th.Tensor],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(texture, output_dir / "texture.png")
    save_image(checkerboard, output_dir / "checkerboard.png")

    names = ("initial", "target", "optimized")
    renders = (initial, target, optimized)
    display_images = tuple(
        _composite_over_checkerboard(rendered, checkerboard) for rendered in renders
    )
    for name, rendered, display in zip(names, renders, display_images, strict=True):
        save_image(display.clamp(0.0, 1.0), output_dir / f"{name}.png")
        save_image(
            rendered[0].clamp(0.0, 1.0), output_dir / f"{name}_premultiplied.png"
        )
        save_image(rendered[1].expand(-1, 3, -1, -1), output_dir / f"{name}_alpha.png")

    color_panels = (
        _make_panel(display_images[0], "INITIAL", initial_vertices, transparent_faces),
        _make_panel(display_images[1], "TARGET", target_vertices, transparent_faces),
        _make_panel(
            display_images[2], "OPTIMIZED", optimized_vertices, transparent_faces
        ),
    )
    _join_panels(color_panels).save(output_dir / "comparison.png")

    alpha_panels = tuple(
        _make_panel(rendered[1].expand(-1, 3, -1, -1), label)
        for rendered, label in zip(
            renders, ("INITIAL ALPHA", "TARGET ALPHA", "OPTIMIZED ALPHA"), strict=True
        )
    )
    _join_panels(alpha_panels).save(output_dir / "comparison_alpha.png")


def run_demo(
    output_dir: Path,
    iterations: int,
    height: int,
    width: int,
    geometry_learning_rate: float,
    opacity_learning_rate: float,
) -> Dict[str, object]:
    if not th.cuda.is_available():
        raise RuntimeError("The fused transparent-render demo requires a CUDA device")
    if iterations <= 0:
        raise ValueError("--iterations must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("--height and --width must be positive")
    if geometry_learning_rate <= 0.0 or opacity_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")

    th.manual_seed(20261105)
    device = th.device("cuda")
    target_vertices, initial_vertices, faces, uv = _make_scene(device, height, width)
    texture = _make_texture(device)
    checkerboard = _make_checkerboard(height, width, device)

    transparent_interior_vertex_ids = th.tensor(
        _TRANSPARENT_INTERIOR_VERTEX_IDS, dtype=th.long, device=device
    )
    target_transparent_opacity = th.tensor(
        [[0.46, 0.58, 0.62, 0.48, 0.56, 0.44, 0.52, 0.66]], device=device
    )
    target_opacity = th.cat(
        (
            th.ones((1, _VERTICES_PER_LAYER), device=device),
            th.zeros((1, _VERTICES_PER_LAYER * 2), device=device),
        ),
        dim=1,
    ).scatter(
        1,
        transparent_interior_vertex_ids[None],
        target_transparent_opacity,
    )

    with th.no_grad():
        target = _render(
            target_vertices,
            faces,
            uv,
            texture,
            target_opacity,
            height,
            width,
        )

    opacity_logits = th.nn.Parameter(th.full((1, 8), -1.5, device=device))
    pixel_scale = th.tensor(
        [width / 256.0, height / 256.0], dtype=th.float32, device=device
    )
    transparent_interior_xy = th.nn.Parameter(
        initial_vertices[:, transparent_interior_vertex_ids, :2] / pixel_scale
    )
    optimizer = th.optim.Adam(
        [
            {"params": [transparent_interior_xy], "lr": geometry_learning_rate},
            {"params": [opacity_logits], "lr": opacity_learning_rate},
        ]
    )

    def current_inputs() -> Tuple[th.Tensor, th.Tensor]:
        scatter_indices = transparent_interior_vertex_ids[None, :, None].expand(
            -1, -1, 2
        )
        current_xy = target_vertices[:, :, :2].scatter(
            1, scatter_indices, transparent_interior_xy * pixel_scale
        )
        vertices = th.cat((current_xy, target_vertices[:, :, 2:]), dim=2)
        opacity = th.cat(
            (
                th.ones((1, _VERTICES_PER_LAYER), device=device),
                th.zeros((1, _VERTICES_PER_LAYER * 2), device=device),
            ),
            dim=1,
        ).scatter(
            1,
            transparent_interior_vertex_ids[None],
            opacity_logits.sigmoid(),
        )
        return vertices, opacity

    def objective(rendered: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        color_loss = th.mean((rendered[0] - target[0]) ** 2)
        alpha_loss = th.mean((rendered[1] - target[1]) ** 2)
        return color_loss + 0.5 * alpha_loss

    with th.no_grad():
        current_vertices, initial_opacity = current_inputs()
        initial = _render(
            current_vertices,
            faces,
            uv,
            texture,
            initial_opacity,
            height,
            width,
        )
        initial_loss = objective(initial).item()

    for iteration in range(iterations):
        current_vertices, opacity = current_inputs()
        rendered = _render(
            current_vertices,
            faces,
            uv,
            texture,
            opacity,
            height,
            width,
        )
        loss = objective(rendered)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % 50 == 0 or iteration == iterations - 1:
            print(f"iteration {iteration:4d}: loss={loss.item():.6e}")

    with th.no_grad():
        optimized_vertices, optimized_opacity = current_inputs()
        optimized = _render(
            optimized_vertices,
            faces,
            uv,
            texture,
            optimized_opacity,
            height,
            width,
        )
        final_loss = objective(optimized).item()
        initial_geometry_error = th.sqrt(
            th.mean(
                th.sum(
                    (
                        initial_vertices[:, transparent_interior_vertex_ids, :2]
                        - target_vertices[:, transparent_interior_vertex_ids, :2]
                    )
                    ** 2,
                    dim=2,
                )
            )
        ).item()
        optimized_geometry_error = th.sqrt(
            th.mean(
                th.sum(
                    (
                        optimized_vertices[:, transparent_interior_vertex_ids, :2]
                        - target_vertices[:, transparent_interior_vertex_ids, :2]
                    )
                    ** 2,
                    dim=2,
                )
            )
        ).item()
        alpha_mae = th.mean(th.abs(optimized[1] - target[1])).item()

    _write_outputs(
        output_dir,
        texture,
        checkerboard,
        initial,
        target,
        optimized,
        initial_vertices,
        target_vertices,
        optimized_vertices,
        (faces[len(faces) // 3 : 2 * len(faces) // 3], faces[2 * len(faces) // 3 :]),
    )

    metrics: Dict[str, object] = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_transparent_geometry_rmse_pixels": initial_geometry_error,
        "optimized_transparent_geometry_rmse_pixels": optimized_geometry_error,
        "optimized_alpha_mae": alpha_mae,
        "target_transparent_interior_opacity": target_transparent_opacity[0].tolist(),
        "optimized_transparent_interior_opacity": optimized_opacity[
            0, transparent_interior_vertex_ids
        ].tolist(),
        "target_accumulated_alpha_max": target[1].max().item(),
        "height": height,
        "width": width,
        "iterations": iterations,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2, sort_keys=True)
        output.write("\n")
    return metrics


def main() -> None:
    args = _parse_args()
    metrics = run_demo(
        args.output_dir,
        args.iterations,
        args.height,
        args.width,
        args.geometry_learning_rate,
        args.opacity_learning_rate,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote demo images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

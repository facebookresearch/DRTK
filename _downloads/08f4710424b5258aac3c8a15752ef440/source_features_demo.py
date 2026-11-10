#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the root LICENSE file.
"""Reproduce the source-feature tutorial figures and video."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from drtk.source_features import sample_feature_stencil, source_feature_stencil
from drtk.transparent_render import render_transparent


def triangulate(points):
    """Ear clipping for the one static simple polygon used by the illustration."""
    points = np.asarray(points, dtype=np.float64)
    order = list(range(len(points)))
    if (
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
        < 0
    ):
        order.reverse()
    faces = []

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    while len(order) > 3:
        for i in range(len(order)):
            a, b, c = order[i - 1], order[i], order[(i + 1) % len(order)]
            pa, pb, pc = points[[a, b, c]]
            if cross(pb - pa, pc - pa) <= 1e-10:
                continue
            inside = False
            for j in order:
                if j in (a, b, c):
                    continue
                p = points[j]
                if (
                    min(
                        cross(pb - pa, p - pa),
                        cross(pc - pb, p - pb),
                        cross(pa - pc, p - pc),
                    )
                    >= 0
                ):
                    inside = True
                    break
            if not inside:
                faces.append((a, b, c))
                order.pop(i)
                break
        else:
            raise ValueError("invalid demonstration polygon")
    faces.append(tuple(order))
    return faces


def scene(size, device):
    cells = 12
    grid = th.linspace(-0.5, size - 0.5, cells + 1, device=device)
    yy, xx = th.meshgrid(grid, grid, indexing="ij")
    zz = 2.6 + 0.16 * th.sin(xx / size * 2 * np.pi) * th.cos(yy / size * np.pi)
    rear = th.stack((xx, yy, zz), -1).reshape(-1, 3)
    faces = []
    for y in range(cells):
        for x in range(cells):
            a = y * (cells + 1) + x
            b = a + 1
            d = a + cells + 1
            c = d + 1
            faces.extend(((a, b, c), (a, c, d)))
    points = (
        np.array(
            [
                (82, 167),
                (78, 112),
                (60, 91),
                (51, 76),
                (56, 68),
                (64, 67),
                (83, 85),
                (79, 40),
                (84, 33),
                (92, 34),
                (98, 43),
                (101, 77),
                (104, 24),
                (111, 20),
                (119, 25),
                (121, 77),
                (129, 37),
                (136, 34),
                (143, 41),
                (141, 84),
                (151, 61),
                (159, 60),
                (165, 68),
                (156, 112),
                (147, 153),
                (130, 169),
            ]
        )
        * size
        / 192
    )
    hand = th.tensor(
        np.column_stack((points, np.full(len(points), 1.6))),
        device=device,
        dtype=th.float32,
    )
    count = rear.shape[0]
    faces += [tuple(c + count for c in f) for f in triangulate(points)]
    v = th.cat((rear, hand))[None]
    vi = th.tensor(faces, device=device, dtype=th.int32)
    uv = (v[:, :, :2] + 0.5) / size
    uv = uv.clone()
    uv[:, :count, 0] *= 0.5
    uv[:, count:, 0] = uv[:, count:, 0] * 0.5 + 0.5
    pixels = th.arange(size, device=device, dtype=th.float32)
    py, px = th.meshgrid(pixels, pixels, indexing="ij")
    gx = (px + 0.5) / size * cells
    gy = (py + 0.5) / size * cells
    ix = gx.long()
    iy = gy.long()
    u = gx - ix
    vv = gy - iy
    second = vv > u
    qf = ((iy * cells + ix) * 2 + second.int()).reshape(1, -1).int()
    sb = th.stack(
        (
            th.where(second, 1 - vv, 1 - u),
            th.where(second, u, u - vv),
            th.where(second, vv - u, vv),
        ),
        -1,
    ).reshape(1, -1, 3)
    ztri = v[0, vi[qf[0]].long(), 2][None]
    qb = sb / ztri
    qb = qb / qb.sum(-1, keepdim=True)
    checker = ((px // 12 + py // 12) % 2)[None]
    waves = (0.5 + 0.5 * th.sin(px / size * 4 * np.pi) * th.cos(py / size * 2 * np.pi))[
        None
    ]
    rear_texture = th.cat(
        (
            0.58 + 0.25 * checker + 0.12 * waves,
            0.17 + 0.32 * checker + 0.09 * waves,
            0.12 + 0.08 * checker + 0.12 * waves,
        )
    )[None]
    hand_texture = th.cat(
        (0.10 + 0.08 * checker, 0.35 + 0.22 * waves, 0.70 + 0.20 * checker)
    )[None]
    return v, vi, uv, count, qf, qb, rear_texture.clamp(0, 1), hand_texture.clamp(0, 1)


def render(v, faces, uv, texture, size):
    return render_transparent(
        v,
        faces,
        uv,
        texture,
        size,
        size,
        background=v.new_tensor([0.035, 0.045, 0.065]),
        rgb_activation="identity",
        alpha_activation="clamp",
        edge_gradients=False,
    )[0]


def image(t):
    return t[0].detach().float().permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def visualize(output, frames, size, device):
    base, faces, uv, count, qf, qb, anchor, hand = scene(size, device)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2), dpi=100)
    fig.patch.set_facecolor("#f5f7fb")
    titles = [
        "Source image",
        "New view · ungated",
        "New view · source-visible",
        "Rear-surface confidence",
    ]
    subtitles = [
        "The hand hides the cloth",
        "Hand features leak onto the cloth",
        "Hidden cloth keeps its anchor",
        "Dark = retain surface context",
    ]
    handles = []
    for ax, title, subtitle in zip(axes, titles, subtitles, strict=False):
        ax.set_title(title, fontsize=14, fontweight="bold", color="#17314b", pad=14)
        handles.append(ax.imshow(np.zeros((size, size, 3))))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(subtitle, fontsize=11, color="#52667a", labelpad=12)
    fig.suptitle(
        "Source visibility protects feature identity",
        fontsize=23,
        fontweight="bold",
        color="#17314b",
        y=0.98,
    )
    fig.text(
        0.5,
        0.045,
        "The same source pixel can belong to different surface queries. "
        "Visibility gates features, not geometry.",
        ha="center",
        fontsize=12,
        color="#52667a",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.85, wspace=0.08)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "24",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "source_features_demo.mp4"),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    stats = []
    with th.no_grad():
        for frame in range(frames):
            phase = frame / max(frames - 1, 1) * 2 * np.pi
            v = base.clone()
            v[:, count:, 0] += np.sin(phase) * size * 0.055
            source = render(v, faces, uv, th.cat((anchor, hand), 3), size)
            indices, weights, _, _ = source_feature_stencil(
                v, faces, qf, qb, size, size
            )
            sampled, confidence = sample_feature_stencil(source, indices, weights)
            sampled = sampled.reshape(1, 3, size, size)
            conf = confidence.reshape(1, 1, size, size)
            refined = sampled + (1 - conf) * anchor
            # The query grid projects to these source pixels in this scene.
            naive = source
            target = v.clone()
            target[:, :, 0] -= (
                (0.46 + 0.11 * np.cos(phase)) * size * 0.92 / target[:, :, 2]
            )
            raw_render = render(target, faces, uv, th.cat((naive, hand), 3), size)
            corrected = render(target, faces, uv, th.cat((refined, hand), 3), size)
            truth = render(target, faces, uv, th.cat((anchor, hand), 3), size)
            heat = plt.get_cmap("viridis")(conf[0, 0].cpu().numpy())[:, :, :3]
            for handle, value in zip(
                handles,
                [image(source), image(raw_render), image(corrected), heat],
                strict=False,
            ):
                handle.set_data(value)
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            process.stdin.write(pixels.tobytes())
            if frame == frames // 4:
                fig.savefig(
                    output / "source_features_demo_poster.png",
                    dpi=150,
                    facecolor=fig.get_facecolor(),
                )
            stats.append(
                {
                    "frame": frame,
                    "ungated_mse": float((raw_render - truth).square().mean()),
                    "visible_mse": float((corrected - truth).square().mean()),
                    "mean_confidence": float(conf.mean()),
                }
            )
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("video encoding failed")
    plt.close(fig)
    return stats


def optimize_edge(output, device):
    size = 32
    v = th.tensor(
        [
            [
                [-0.5, -0.5, 1],
                [10.2, -0.5, 1],
                [10.2, 31.5, 1],
                [-0.5, 31.5, 1],
                [-0.5, -0.5, 2],
                [31.5, -0.5, 2],
                [31.5, 31.5, 2],
                [-0.5, 31.5, 2],
            ]
        ],
        device=device,
    )
    faces = th.tensor(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], device=device, dtype=th.int32
    )
    yy, xx = th.meshgrid(
        th.arange(2, 30, device=device), th.arange(2, 30, device=device), indexing="ij"
    )
    x = (xx.float() + 0.5) / 32
    y = (yy.float() + 0.5) / 32
    second = y > x
    qb = th.stack(
        (
            th.where(second, 1 - y, 1 - x),
            th.where(second, x, x - y),
            th.where(second, y - x, y),
        ),
        -1,
    ).reshape(1, -1, 3)
    qf = (2 + second.int()).reshape(1, -1)
    features = th.ones((1, 3, size, size), device=device)
    movement = th.zeros_like(v)
    movement[0, [1, 2], 0] = 1
    target_shift = 7.0
    with th.no_grad():
        idx, w, _, _ = source_feature_stencil(
            v + target_shift * movement, faces, qf, qb, size, size
        )
        desired, _ = sample_feature_stencil(features, idx, w)
    histories = {}
    results = {}
    for enabled in [False, True]:
        shift = th.tensor(0.0, device=device, requires_grad=True)
        optimizer = th.optim.Adam([shift], lr=0.22)
        history = []
        for step in range(100):
            optimizer.zero_grad()
            idx, w, _, _ = source_feature_stencil(
                v + shift * movement, faces, qf, qb, size, size, edge_gradients=enabled
            )
            sample, _ = sample_feature_stencil(features, idx, w)
            loss = (sample - desired).square().mean()
            loss.backward()
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "edge_x": 10.2 + float(shift.detach()),
                }
            )
        histories["edge_grad" if enabled else "smooth_only"] = history
        results[enabled] = sample.detach()[0, 0].reshape(28, 28).cpu().numpy()
    fig = plt.figure(figsize=(14, 7), dpi=130, facecolor="#f5f7fb")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    for col, (name, data) in enumerate(
        [
            ("Target", desired[0, 0].reshape(28, 28).cpu().numpy()),
            ("Smooth backward only", results[False]),
            ("Automatic edge gradients", results[True]),
        ]
    ):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(data, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(name, color="#17314b", fontweight="bold")
        ax.set_axis_off()
    ax = fig.add_subplot(grid[1, :2])
    bx = fig.add_subplot(grid[1, 2])
    for key, label, color in [
        ("smooth_only", "Smooth only", "#d66a41"),
        ("edge_grad", "With edge gradients", "#167d9a"),
    ]:
        h = histories[key]
        ax.plot(
            [a["step"] for a in h],
            [a["edge_x"] for a in h],
            label=label,
            color=color,
            lw=2.5,
        )
        bx.plot([a["step"] for a in h], [a["loss"] for a in h], color=color, lw=2.5)
    ax.axhline(17.2, color="#52667a", linestyle="--", lw=1, label="Target edge")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Occluder edge (source pixels)")
    ax.legend(frameon=False)
    bx.set_xlabel("Optimization step")
    bx.set_ylabel("Sampled-feature MSE")
    for a in [ax, bx]:
        a.grid(alpha=0.15)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "A feature loss can move the occluding geometry",
        fontsize=21,
        color="#17314b",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(
        output / "source_features_edge_optimization.png",
        dpi=150,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("source_features_demo"))
    parser.add_argument("--frames", type=int, default=144)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    th.manual_seed(12)
    stats = visualize(args.output, args.frames, args.size, args.device)
    optimization = optimize_edge(args.output, args.device)
    (args.output / "source_features_demo.json").write_text(
        json.dumps({"frames": stats, "optimization": optimization}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

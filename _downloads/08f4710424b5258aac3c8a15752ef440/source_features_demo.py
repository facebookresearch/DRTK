#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the root LICENSE file.
"""Fit deformation and opacity from transmittance in an original surface grid."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from drtk.source_features import source_feature_stencil
from drtk.transparent_render import render_transparent
from matplotlib.ticker import NullFormatter


def to_image(tensor):
    return tensor.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()


class Scene:
    def __init__(self, size, device):
        self.size = size
        self.texture_width, self.texture_height = 128, 64
        nu, nv = 80, 24
        yy, xx = th.meshgrid(
            th.linspace(0, 1, nv + 1, device=device),
            th.linspace(0, 1, nu + 1, device=device),
            indexing="ij",
        )
        corners = th.tensor(
            [
                [-1.1, 0.4],
                [-0.55, 0.05],
                [0.6, 0.50],
                [-0.10, -0.20],
                [0.9, -0.38],
                [1.15, 0.15],
            ],
            device=device,
        )
        position = xx * 5
        segment = position.long().clamp_max(4)
        fraction = position - segment
        xz = (
            corners[segment] * (1 - fraction[..., None])
            + corners[segment + 1] * fraction[..., None]
        )
        self.world = th.stack((xz[..., 0], (yy - 0.5) * 1.65, xz[..., 1]), -1).reshape(
            -1, 3
        )
        self.uv = th.stack((xx, yy), -1).reshape(1, -1, 2)
        faces = []
        for y in range(nv):
            for x in range(nu):
                a = y * (nu + 1) + x
                b, d = a + 1, a + nu + 1
                c = d + 1
                faces.extend(((a, b, c), (a, c, d)))
        self.faces = th.tensor(faces, device=device, dtype=th.int32)
        v, u = th.meshgrid(
            (th.arange(self.texture_height, device=device) + 0.5) / self.texture_height,
            (th.arange(self.texture_width, device=device) + 0.5) / self.texture_width,
            indexing="ij",
        )
        texture = th.stack(
            (
                0.48 + 0.29 * th.sin(2 * th.pi * u + 0.6),
                0.48 + 0.27 * th.sin(2 * th.pi * u + 2.7),
                0.48 + 0.25 * th.sin(2 * th.pi * u + 4.8),
            ),
            0,
        )
        checker = ((u * 16).long() + (v * 10).long()) % 2
        self.texture = (
            texture + (0.16 * checker - 0.08)[None] + 0.08 * th.sin(v * 2 * th.pi)[None]
        ).clamp(0, 1)[None]
        gx, gy = u * nu, v * nv
        ix, iy = gx.long(), gy.long()
        du, dv = gx - ix, gy - iy
        second = dv > du
        self.query_faces = ((iy * nu + ix) * 2 + second.int()).reshape(1, -1).int()
        self.query_barycentrics = th.stack(
            (
                th.where(second, 1 - dv, 1 - du),
                th.where(second, du, du - dv),
                th.where(second, dv - du, dv),
            ),
            -1,
        ).reshape(1, -1, 3)
        self.source_angles = [-65, 0, 65]
        self.corners = corners
        self.fold_delta = corners.new_tensor([[-0.18, 0.16], [0.18, 0.13]])

    def deform(self, amount):
        corners = th.cat(
            (
                self.corners[:2],
                self.corners[2:4] + amount * self.fold_delta,
                self.corners[4:],
            )
        )
        position = self.uv[0, :, 0] * 5
        segment = position.long().clamp_max(4)
        fraction = (position - segment)[:, None]
        xz = corners[segment] * (1 - fraction) + corners[segment + 1] * fraction
        return th.stack((xz[:, 0], self.world[:, 1], xz[:, 1]), -1)

    def project(self, angles, world=None):
        world = self.world if world is None else world
        views = []
        for degrees in angles:
            angle = np.deg2rad(degrees)
            x = np.cos(angle) * world[:, 0] + np.sin(angle) * world[:, 2]
            z = -np.sin(angle) * world[:, 0] + np.cos(angle) * world[:, 2] + 3.2
            focal = self.size * (210 / 192)
            views.append(
                th.stack(
                    (
                        x / z * focal + (self.size - 1) / 2,
                        world[:, 1] / z * focal + (self.size - 1) / 2,
                        z,
                    ),
                    -1,
                )
            )
        return th.stack(views)

    def render(self, vertices, alpha):
        count = vertices.shape[0]
        return render_transparent(
            vertices,
            self.faces,
            self.uv.expand(count, -1, -1),
            self.texture.expand(count, -1, -1, -1),
            self.size,
            self.size,
            opacity=alpha.expand(count, vertices.shape[1]),
            background=vertices.new_tensor([0.94, 0.95, 0.97]),
            rgb_activation="identity",
            alpha_activation="clamp",
            edge_gradients=False,
        )[0]


def transmittance(scene, amount, alpha, edge_gradients=True):
    vertices = scene.project(scene.source_angles, scene.deform(amount))
    _, weights, _, _ = source_feature_stencil(
        vertices,
        scene.faces,
        scene.query_faces.expand(3, -1),
        scene.query_barycentrics.expand(3, -1, -1),
        scene.size,
        scene.size,
        vertex_opacity=alpha.expand(3, vertices.shape[1]),
        edge_gradients=edge_gradients,
    )
    return weights.sum(-1).reshape(3, scene.texture_height, scene.texture_width)


def fit(scene, steps):
    zero = scene.world.new_tensor(0.0)
    target_alpha = scene.world.new_tensor(0.65)
    with th.no_grad():
        target = transmittance(scene, zero, target_alpha, False)
    histories, fields, images = {}, [], []
    for edge_gradients in (False, True):
        amount = scene.world.new_tensor(0.6, requires_grad=True)
        logit = scene.world.new_tensor(np.log(0.4 / 0.6), requires_grad=True)
        optimizer = th.optim.Adam(
            [
                {"params": [amount], "lr": 0.015},
                {"params": [logit], "lr": 0.04},
            ]
        )
        history = []
        for step in range(steps + 1):
            optimizer.zero_grad()
            alpha = logit.sigmoid()
            prediction = transmittance(scene, amount, alpha, edge_gradients)
            loss = (prediction - target).square().mean()
            if step % 4 == 0 or step == steps:
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "fold": float(amount.detach()),
                        "alpha": float(alpha.detach()),
                    }
                )
                if edge_gradients:
                    fields.append(prediction.detach().cpu().numpy())
                    with th.no_grad():
                        vertices = scene.project(
                            scene.source_angles, scene.deform(amount)
                        )
                        render = scene.render(vertices, alpha)
                        images.append(
                            (render.clamp(0, 1) * 255)
                            .byte()
                            .permute(0, 2, 3, 1)
                            .cpu()
                            .numpy()
                        )
            if step < steps:
                loss.backward()
                optimizer.step()
                if step + 1 in (steps // 2, 3 * steps // 4):
                    for group in optimizer.param_groups:
                        group["lr"] *= 0.25
        histories["boundary" if edge_gradients else "smooth"] = history
    return histories, np.asarray(fields), np.asarray(images), target.cpu().numpy()


def clean_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def same_pixel_queries(scene):
    uv = scene.world.new_tensor(
        [[0.62, 0.5], [(2 + 6 / 7) / 5, 0.5], [(1 + 0.55 / 1.15) / 5, 0.5]]
    )
    gx, gy = uv[:, 0] * 80, uv[:, 1] * 24
    ix, iy = gx.long(), gy.long()
    du, dv = gx - ix, gy - iy
    second = dv > du
    face_ids = ((iy * 80 + ix) * 2 + second.int())[None].int()
    barycentrics = th.stack(
        (
            th.where(second, 1 - dv, 1 - du),
            th.where(second, du, du - dv),
            th.where(second, dv - du, dv),
        ),
        -1,
    )[None]
    vertices = scene.project([0])
    alpha = scene.world.new_tensor(0.65, requires_grad=True)
    _, weights, projected, _ = source_feature_stencil(
        vertices,
        scene.faces,
        face_ids,
        barycentrics,
        scene.size,
        scene.size,
        vertex_opacity=alpha.expand(1, vertices.shape[1]),
        return_queries=True,
    )
    values = weights.sum(-1)[0]
    derivatives = [
        float(th.autograd.grad(value, alpha, retain_graph=True)[0]) for value in values
    ]
    th.testing.assert_close(
        projected[0], th.zeros_like(projected[0]), atol=2e-5, rtol=0
    )
    th.testing.assert_close(
        values, values.new_tensor([1, 0.35, 0.1225]), atol=2e-5, rtol=0
    )
    th.testing.assert_close(
        values.new_tensor(derivatives),
        values.new_tensor([0, -1, -0.7]),
        atol=2e-5,
        rtol=0,
    )
    return uv.cpu().numpy(), values.detach().cpu().tolist(), derivatives


def draw_pullback(output, scene, target):
    uv, values, derivatives = same_pixel_queries(scene)
    with th.no_grad():
        image = to_image(
            scene.render(scene.project([0]), scene.world.new_tensor(0.65))[0]
        )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8), dpi=100, facecolor="#f5f7fb")
    axes[0].imshow(
        to_image(scene.texture[0]), extent=(0, 1, 1, 0), interpolation="nearest"
    )
    axes[1].imshow(image, interpolation="nearest")
    axes[2].imshow(
        target[1],
        extent=(0, 1, 1, 0),
        vmin=0,
        vmax=1,
        cmap="viridis",
        interpolation="nearest",
    )
    for i, title in enumerate(
        [
            "Original surface grid",
            "Deformed surface in the camera",
            "Transmittance in the original grid",
        ]
    ):
        axes[i].set_title(title, fontsize=17, color="#17314b", weight="bold", pad=18)
        clean_axis(axes[i])
    for i, (u, v) in enumerate(uv):
        for axis in (axes[0], axes[2]):
            axis.plot(u, v, "o", color="white", mec="#17314b", ms=7, zorder=5)
            axis.annotate(
                f"q{i + 1}",
                (u, v),
                (u, 0.1 + i * 0.13),
                ha="center",
                fontsize=13,
                color="#17314b",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
                arrowprops={"arrowstyle": "-", "color": "white", "lw": 1.4},
            )
    center = (scene.size - 1) / 2
    axes[1].plot(center, center, "+", color="white", ms=15, mew=2)
    axes[1].annotate(
        "p", (center, center), (center + 12, center - 12), color="white", fontsize=18
    )
    subtitles = [
        "Each point keeps its original index",
        "q1, q2 and q3 project to pixel p",
        "Each point receives its own value",
    ]
    for i, text in enumerate(subtitles):
        fig.text((i + 0.5) / 3, 0.21, text, ha="center", fontsize=15, color="#52667a")
    fig.suptitle(
        "Transmittance returns to the original grid",
        fontsize=27,
        color="#17314b",
        weight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.89,
        "Several surface points can share a camera pixel while retaining different transmittances.",
        ha="center",
        fontsize=16,
        color="#52667a",
    )
    for x in (1 / 3, 2 / 3):
        fig.text(x, 0.51, "→", ha="center", fontsize=30, color="#52667a")
    labels = [r"$T(q_1)=1$", r"$T(q_2)=1-\alpha=0.35$", r"$T(q_3)=(1-\alpha)^2=0.1225$"]
    for i, label in enumerate(labels):
        fig.text((i + 0.5) / 3, 0.115, label, ha="center", fontsize=20, color="#17314b")
    fig.text(
        0.5,
        0.045,
        r"Material opacity $\alpha=0.65$. The receiving surface's own opacity is excluded.",
        ha="center",
        fontsize=14,
        color="#52667a",
    )
    fig.subplots_adjust(left=0.025, right=0.975, top=0.79, bottom=0.27, wspace=0.12)
    fig.savefig(output / "source_features_sampling.png", dpi=100)
    plt.close(fig)
    return {
        "query_uv": uv.tolist(),
        "transmittance": values,
        "opacity_derivative": derivatives,
    }


def make_fit_figure(scene, histories, fields, images, target):
    fig = plt.figure(figsize=(16, 13), dpi=100, facecolor="#f5f7fb")
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=[2.5, 1.1, 1.1, 1.1],
        left=0.035,
        right=0.965,
        top=0.88,
        bottom=0.12,
        hspace=0.34,
        wspace=0.08,
    )
    render_handles, field_handles = [], []
    for view in range(3):
        axis = fig.add_subplot(grid[0, view])
        render_handles.append(axis.imshow(images[0, view], interpolation="nearest"))
        axis.set_title(
            f"Source camera {'ABC'[view]}", fontsize=21, color="#17314b", weight="bold"
        )
        clean_axis(axis)
        for row, data in ((1, target[view]), (2, fields[0, view])):
            axis = fig.add_subplot(grid[row, view])
            handle = axis.imshow(
                data, cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
            )
            axis.set_title(
                "Target · original grid" if row == 1 else "Estimate · original grid",
                fontsize=15,
                color="#52667a",
            )
            clean_axis(axis)
            if row == 2:
                field_handles.append(handle)
    color_axis = fig.add_axes((0.37, 0.035, 0.26, 0.012))
    colorbar = fig.colorbar(
        field_handles[0],
        cax=color_axis,
        orientation="horizontal",
        ticks=[0, 0.35, 0.65, 1],
    )
    colorbar.ax.set_title("Transmittance", fontsize=11, pad=6)
    chart = fig.add_subplot(grid[3, 1:])
    lines = [
        chart.plot([], [], color=color, lw=2.5, label=label)[0]
        for color, label in [
            ("#127f9b", "With boundary gradients"),
            ("#c66b39", "Smooth gradients only"),
        ]
    ]
    minimum = min(row["loss"] for values in histories.values() for row in values)
    maximum = max(row["loss"] for values in histories.values() for row in values)
    chart.set(
        yscale="log",
        xlim=(0, histories["boundary"][-1]["step"]),
        ylim=(max(minimum * 0.7, 1e-7), maximum * 1.4),
        xlabel="Optimization step",
        ylabel="Original-grid MSE",
    )
    chart.set_yticks(
        [0.002, 0.005, 0.01, 0.02, 0.04], ["0.002", "0.005", "0.01", "0.02", "0.04"]
    )
    chart.yaxis.set_minor_formatter(NullFormatter())
    chart.spines[["top", "right"]].set_visible(False)
    chart.grid(alpha=0.18)
    chart.legend(frameon=False, fontsize=12, loc="upper right")
    status = fig.add_subplot(grid[3, 0])
    status.axis("off")
    step_text = status.text(0, 0.78, "", fontsize=22, color="#17314b", weight="bold")
    alpha_text = status.text(0, 0.46, "", fontsize=15, color="#52667a")
    fold_text = status.text(0, 0.16, "", fontsize=15, color="#52667a")
    fig.suptitle(
        "Fitting transmittance in the original grid",
        fontsize=28,
        color="#17314b",
        weight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.928,
        "Gradients from the returned field update the fold and its material opacity.",
        ha="center",
        fontsize=17,
        color="#52667a",
    )

    def update(index):
        row = histories["boundary"][index]
        for view in range(3):
            render_handles[view].set_data(images[index, view])
            field_handles[view].set_data(fields[index, view])
        for line, name in zip(lines, ["boundary", "smooth"], strict=True):
            partial = histories[name][: index + 1]
            line.set_data(
                [item["step"] for item in partial], [item["loss"] for item in partial]
            )
        step_text.set_text(f"Step {row['step']} / {histories['boundary'][-1]['step']}")
        alpha_text.set_text(f"Opacity: {row['alpha']:.3f}  (target 0.650)")
        fold_text.set_text(f"Fold offset: {row['fold']:.3f}  (target 0.000)")
        fig.canvas.draw()

    return fig, update


def write_video(output, scene, histories, fields, images, target, fps):
    fig, update = make_fit_figure(scene, histories, fields, images, target)
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
        "1600x1300",
        "-r",
        str(fps),
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
    assert process.stdin is not None
    history = histories["boundary"]
    stride = max(1, len(history) // 10)
    for index in range(len(fields)):
        early = history[index]["step"] <= 0.3 * history[-1]["step"]
        if not early and index % stride != 0 and index != len(fields) - 1:
            continue
        update(index)
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().tobytes()
        repeats = 2 * fps if index in (0, len(fields) - 1) else (3 if early else 1)
        for _ in range(repeats):
            process.stdin.write(pixels)
        if index in (0, len(fields) // 2, len(fields) - 1):
            fig.savefig(
                output
                / f"source_features_step_{histories['boundary'][index]['step']:03d}.png",
                dpi=100,
            )
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("video encoding failed")
    fig.savefig(output / "source_features_demo_poster.png", dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("source_features_demo"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 4 or args.size < 32 or args.fps < 1:
        parser.error("steps must be at least 4, size at least 32, and fps positive")
    args.output.mkdir(parents=True, exist_ok=True)
    scene = Scene(args.size, args.device)
    histories, fields, images, target = fit(scene, args.steps)
    queries = draw_pullback(args.output, scene, target)
    write_video(args.output, scene, histories, fields, images, target, args.fps)
    (args.output / "source_features_demo.json").write_text(
        json.dumps(
            {
                "source_angles": scene.source_angles,
                "steps": args.steps,
                "size": args.size,
                "same_pixel_queries": queries,
                "histories": histories,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        args.output / "source_features_fit.npz",
        fields=fields,
        images=images,
        target=target,
    )
    print(
        json.dumps({name: values[-1] for name, values in histories.items()}, indent=2)
    )


if __name__ == "__main__":
    main()

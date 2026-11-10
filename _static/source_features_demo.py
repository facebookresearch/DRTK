#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the root LICENSE file.
"""Fit a folded surface's appearance from three partially occluded views."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn.functional as F
from drtk.source_features import sample_feature_stencil, source_feature_stencil
from drtk.transparent_render import render_transparent


def to_image(tensor):
    return tensor.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def to_bytes(tensor):
    return (tensor.detach().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


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
        self.coordinates = th.stack((u, v), -1).reshape(-1, 2)
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
        self.vertices = self.project(self.source_angles)
        self.held_vertices = self.project([35])
        with th.no_grad():
            self.images = self.render(self.vertices, self.texture)
            indices, weights, query_uv, _ = source_feature_stencil(
                self.vertices,
                self.faces,
                self.query_faces.expand(3, -1),
                self.query_barycentrics.expand(3, -1, -1),
                size,
                size,
                return_queries=True,
            )
            self.samples, self.confidence = sample_feature_stencil(
                self.images, indices, weights
            )
            self.plain = F.grid_sample(
                self.images, query_uv[:, None], align_corners=False
            ).squeeze(2)
            self.in_image = (query_uv.abs() <= 1).all(-1)[:, None].float()
            self.target = self.render(self.held_vertices, self.texture)[0]

    def project(self, angles):
        views = []
        for degrees in angles:
            angle = np.deg2rad(degrees)
            x = np.cos(angle) * self.world[:, 0] + np.sin(angle) * self.world[:, 2]
            z = (
                -np.sin(angle) * self.world[:, 0]
                + np.cos(angle) * self.world[:, 2]
                + 3.2
            )
            focal = self.size * (210 / 192)
            views.append(
                th.stack(
                    (
                        x / z * focal + (self.size - 1) / 2,
                        self.world[:, 1] / z * focal + (self.size - 1) / 2,
                        z,
                    ),
                    -1,
                )
            )
        return th.stack(views)

    def render(self, vertices, textures):
        count = vertices.shape[0]
        return render_transparent(
            vertices,
            self.faces,
            self.uv.expand(count, -1, -1),
            textures.expand(count, -1, -1, -1),
            self.size,
            self.size,
            background=vertices.new_tensor([0.94, 0.95, 0.97]),
            rgb_activation="identity",
            alpha_activation="clamp",
            edge_gradients=False,
        )[0]


def fit_appearance(scene, steps):
    coordinates = scene.coordinates
    frequency = coordinates.new_tensor([1, 2, 4, 8, 16]) * (2 * th.pi)
    phase = (coordinates[:, :, None] * frequency).flatten(1)
    inputs = th.cat((coordinates, phase.sin(), phase.cos()), 1)
    th.manual_seed(5)
    model = th.nn.Sequential(
        th.nn.Linear(inputs.shape[1], 96),
        th.nn.ReLU(),
        th.nn.Linear(96, 96),
        th.nn.ReLU(),
        th.nn.Linear(96, 3),
        th.nn.Sigmoid(),
    ).to(coordinates.device)
    models = [model, copy.deepcopy(model)]
    optimizers = [th.optim.Adam(model.parameters(), lr=0.003) for model in models]
    weights = [scene.in_image, scene.confidence]
    observations = [scene.plain, scene.samples / scene.confidence.clamp_min(1e-6)]
    history, renders, textures = [], [], []
    for step in range(steps + 1):
        values, losses = [], []
        for model, optimizer, observed, weight in zip(
            models, optimizers, observations, weights, strict=True
        ):
            optimizer.zero_grad()
            prediction = model(inputs).T[None]
            loss = ((prediction - observed).square() * weight).sum() / (
                3 * weight.sum()
            )
            values.append(
                prediction.detach().reshape(
                    1, 3, scene.texture_height, scene.texture_width
                )
            )
            losses.append(float(loss.detach()))
            if step < steps:
                loss.backward()
                optimizer.step()
        if step % 4 == 0 or step == steps:
            with th.no_grad():
                texture = th.cat(values)
                rendered = scene.render(scene.held_vertices.expand(2, -1, -1), texture)
                mse = (rendered - scene.target[None]).square().mean((1, 2, 3))
                texture_mse = (texture - scene.texture).square().mean((1, 2, 3))
                history.append(
                    {
                        "step": step,
                        "training_loss": losses,
                        "heldout_mse": mse.cpu().tolist(),
                        "texture_mse": texture_mse.cpu().tolist(),
                    }
                )
                renders.append(to_bytes(rendered))
                textures.append(to_bytes(texture))
    return history, np.asarray(renders), np.asarray(textures)


def clean_image_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_sources(output, scene):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 8),
        dpi=100,
        gridspec_kw={"height_ratios": [2, 1]},
        facecolor="#f5f7fb",
    )
    for index in range(3):
        axes[0, index].imshow(to_image(scene.images[index]), interpolation="nearest")
        axes[0, index].set_title(
            f"Source camera {'ABC'[index]}", fontsize=22, color="#17314b", weight="bold"
        )
        axes[1, index].imshow(
            scene.confidence[index, 0]
            .reshape(scene.texture_height, scene.texture_width)
            .cpu(),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[1, index].set_title(
            "Visibility on the unfolded surface", fontsize=15, color="#52667a"
        )
        clean_image_axis(axes[0, index])
        clean_image_axis(axes[1, index])
    fig.suptitle(
        "Different cameras see different folds",
        fontsize=28,
        color="#17314b",
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "White: visible from this camera · Black: occluded",
        ha="center",
        fontsize=16,
        color="#52667a",
    )
    fig.subplots_adjust(
        left=0.025, right=0.975, top=0.88, bottom=0.10, hspace=0.24, wspace=0.08
    )
    fig.savefig(output / "source_features_sampling.png", dpi=100)
    plt.close(fig)


def make_fit_figure(scene, history, renders, textures):
    fig = plt.figure(figsize=(16, 10), dpi=100, facecolor="#f5f7fb")
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[2.8, 1.2, 1.05],
        left=0.035,
        right=0.965,
        top=0.86,
        bottom=0.08,
        hspace=0.28,
        wspace=0.09,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    atlas_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    titles = ["Held-out target", "Ordinary projection", "Source visibility"]
    image_handles, texture_handles = [], []
    for index in range(3):
        value = to_image(scene.target) if index == 0 else renders[0, index - 1]
        atlas = to_image(scene.texture[0]) if index == 0 else textures[0, index - 1]
        image_handles.append(axes[index].imshow(value, interpolation="nearest"))
        texture_handles.append(atlas_axes[index].imshow(atlas, interpolation="nearest"))
        axes[index].set_title(
            titles[index], fontsize=22, color="#17314b", weight="bold", pad=15
        )
        atlas_axes[index].set_title(
            "Surface texture" if index == 0 else "Fitted texture",
            fontsize=15,
            color="#52667a",
        )
        clean_image_axis(axes[index])
        clean_image_axis(atlas_axes[index])
    chart = fig.add_subplot(grid[2, 1:])
    colors = ["#c66b39", "#127f9b"]
    lines = [
        chart.plot([], [], color=color, lw=2.5, label=title)[0]
        for color, title in zip(colors, titles[1:], strict=True)
    ]
    scores = -10 * np.log10(
        np.maximum(np.array([row["heldout_mse"] for row in history]), 1e-12)
    )
    chart.set(
        xlim=(0, history[-1]["step"]),
        ylim=(np.floor(scores.min()) - 1, np.ceil(scores.max()) + 1),
        xlabel="Optimization step",
        ylabel="Held-out PSNR (dB)",
    )
    chart.spines[["top", "right"]].set_visible(False)
    chart.grid(alpha=0.18)
    chart.legend(frameon=False, loc="lower right", fontsize=11)
    status = fig.add_subplot(grid[2, 0])
    status.axis("off")
    step_text = status.text(0, 0.75, "", fontsize=21, color="#17314b", weight="bold")
    score_text = [
        status.text(0, y, "", fontsize=14, color=color)
        for y, color in zip([0.40, 0.13], colors, strict=True)
    ]
    fig.suptitle(
        "Learning appearance from occluded views",
        fontsize=28,
        color="#17314b",
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Fit a texture from three source images; evaluate it from a fourth camera.",
        ha="center",
        fontsize=16,
        color="#52667a",
    )

    def update(index):
        step = history[index]["step"]
        for model in range(2):
            image_handles[model + 1].set_data(renders[index, model])
            texture_handles[model + 1].set_data(textures[index, model])
            lines[model].set_data(
                [row["step"] for row in history[: index + 1]],
                scores[: index + 1, model],
            )
            score_text[model].set_text(
                f"{titles[model + 1]}: {scores[index, model]:.1f} dB"
            )
        step_text.set_text(f"Step {step} / {history[-1]['step']}")
        fig.canvas.draw()

    return fig, update


def write_video(output, scene, history, renders, textures, fps):
    fig, update = make_fit_figure(scene, history, renders, textures)
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
        "1600x1000",
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
    for index in range(len(history)):
        update(index)
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().tobytes()
        repeat = 2 * fps if index in (0, len(history) - 1) else 1
        for _ in range(repeat):
            process.stdin.write(pixels)
        if index in (0, len(history) // 2, len(history) - 1):
            fig.savefig(
                output / f"source_features_step_{history[index]['step']:03d}.png",
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
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.size < 16 or args.fps < 1:
        parser.error("steps and fps must be positive; size must be at least 16")
    args.output.mkdir(parents=True, exist_ok=True)
    scene = Scene(args.size, args.device)
    history, renders, textures = fit_appearance(scene, args.steps)
    draw_sources(args.output, scene)
    write_video(args.output, scene, history, renders, textures, args.fps)
    (args.output / "source_features_demo.json").write_text(
        json.dumps(
            {
                "source_angles": scene.source_angles,
                "heldout_angle": 35,
                "steps": args.steps,
                "size": args.size,
                "observed_surface_fraction": float(
                    (scene.confidence.sum(0) > 0).float().mean()
                ),
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        args.output / "source_features_fit.npz", renders=renders, textures=textures
    )
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()

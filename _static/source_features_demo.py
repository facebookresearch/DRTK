#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the root LICENSE file.
"""Fit an occluding polygon using a sampled-feature loss and record the fit."""

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


def as_image(tensor):
    return tensor[0].detach().float().permute(1, 2, 0).cpu().numpy()


class Scene:
    def __init__(self, size, device):
        self.size = size
        angles = th.arange(20, device=device) * (2 * th.pi / 20)
        self.directions = th.stack((th.cos(angles), th.sin(angles)), -1)
        target_radii = (
            size * 0.27 * (1 + 0.18 * th.cos(3 * angles) - 0.10 * th.sin(2 * angles))
        )
        target_center = th.tensor([0.54, 0.48], device=device) * size - 0.5
        self.target = target_center + self.directions * target_radii[:, None]
        self.initial_center = th.tensor([0.43, 0.56], device=device) * size - 0.5
        self.initial_radii = th.full((20,), 0.22 * size, device=device)
        self.initial = (
            self.initial_center + self.directions * self.initial_radii[:, None]
        )
        self.rear = th.tensor(
            [
                [-0.5, -0.5, 2],
                [size - 0.5, -0.5, 2],
                [size - 0.5, size - 0.5, 2],
                [-0.5, size - 0.5, 2],
            ],
            device=device,
        )
        self.faces = th.tensor(
            [[0, 1, 2], [0, 2, 3]] + [[4, 5 + i, 5 + (i + 1) % 20] for i in range(20)],
            dtype=th.int32,
            device=device,
        )
        pixels = th.arange(size, device=device)
        yy, xx = th.meshgrid(pixels, pixels, indexing="ij")
        x, y = (xx + 0.5) / size, (yy + 0.5) / size
        second = y > x
        self.query_faces = second.int().reshape(1, -1)
        self.query_barycentrics = th.stack(
            (
                th.where(second, 1 - y, 1 - x),
                th.where(second, x, x - y),
                th.where(second, y - x, y),
            ),
            -1,
        ).reshape(1, -1, 3)
        checker = ((xx // (size // 8) + yy // (size // 8)) % 2).float()
        pattern = th.stack(
            (0.70 + 0.20 * checker, 0.36 + 0.18 * checker, 0.12 + 0.10 * checker),
            0,
        )[None]
        blue = th.tensor([0.08, 0.48, 0.76], device=device)
        self.texture = th.cat(
            (pattern, blue[None, :, None, None].expand_as(pattern)), 3
        )
        self.uv = (self.vertices(self.target)[..., :2] + 0.5) / size
        self.uv[:, :4, 0] *= 0.5
        self.uv[:, 4:, 0] = 0.75
        with th.no_grad():
            self.source = self.render(self.target)
            self.desired, self.target_confidence = self.sample(self.target)

    def vertices(self, boundary):
        front = th.cat((boundary.mean(0, keepdim=True), boundary), 0)
        front = th.cat((front, th.ones_like(front[:, :1])), 1)
        return th.cat((self.rear, front), 0)[None]

    def render(self, boundary):
        return render_transparent(
            self.vertices(boundary),
            self.faces,
            self.uv,
            self.texture,
            self.size,
            self.size,
            rgb_activation="identity",
            alpha_activation="clamp",
            edge_gradients=False,
        )[0]

    def sample(self, boundary, edge_gradients=True):
        indices, weights, _, _ = source_feature_stencil(
            self.vertices(boundary),
            self.faces,
            self.query_faces,
            self.query_barycentrics,
            self.size,
            self.size,
            edge_gradients=edge_gradients,
        )
        return sample_feature_stencil(self.source, indices, weights)


def fit(scene, steps, edge_gradients):
    center = scene.initial_center.clone().requires_grad_()
    radii = scene.initial_radii.clone().requires_grad_()
    optimizer = th.optim.Adam([center, radii], lr=0.28 * scene.size / 128)
    history, boundaries, images, errors = [], [], [], []
    for step in range(steps + 1):
        optimizer.zero_grad()
        optimizer.param_groups[0]["lr"] = (
            0.28
            * scene.size
            / 128
            * (0.04 + 0.96 * (0.5 + 0.5 * np.cos(np.pi * step / steps)))
        )
        boundary = (
            center + scene.directions * radii.clamp_min(0.01 * scene.size)[:, None]
        )
        sampled, confidence = scene.sample(boundary, edge_gradients=edge_gradients)
        loss = (sampled - scene.desired).square().mean()
        with th.no_grad():
            history.append(
                {
                    "step": step,
                    "feature_mse": float(loss),
                    "visibility_disagreement": float(
                        (confidence - scene.target_confidence).abs().mean()
                    ),
                }
            )
            if edge_gradients:
                boundaries.append(boundary.cpu().numpy().copy())
                images.append(as_image(scene.render(boundary)))
                errors.append(
                    (sampled - scene.desired)
                    .abs()
                    .mean(1)
                    .reshape(scene.size, scene.size)
                    .cpu()
                    .numpy()
                )
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    return history, np.asarray(boundaries), np.asarray(images), np.asarray(errors)


def draw_outputs(output, scene, history, control, boundaries, images, errors, fps):
    steps = len(history) - 1
    initial_loss = history[0]["feature_mse"]
    losses = np.array([item["feature_mse"] for item in history]) / initial_loss * 100
    control_losses = (
        np.array([item["feature_mse"] for item in control]) / initial_loss * 100
    )
    target = scene.target.cpu().numpy()
    target_loop = np.concatenate((target, target[:1]))
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#f5f7fb")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[3.2, 1],
        left=0.035,
        right=0.965,
        bottom=0.09,
        top=0.845,
        hspace=0.35,
        wspace=0.10,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    chart = fig.add_subplot(grid[1, 1:])
    status = fig.add_subplot(grid[1, 0])
    status.axis("off")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].imshow(as_image(scene.source), interpolation="nearest")
    axes[0].set_title("Target", fontsize=22, color="#17314b", pad=15, weight="bold")
    axes[0].set_xlabel(
        "Blue polygon in front of a textured plane", fontsize=12, labelpad=12
    )
    current = axes[1].imshow(images[0], interpolation="nearest")
    axes[1].set_title(
        "Current fit", fontsize=22, color="#17314b", pad=15, weight="bold"
    )
    axes[1].set_xlabel(
        "White dots: fitted vertices · dashed: target", fontsize=12, labelpad=12
    )
    axes[1].plot(*target_loop.T, color="white", ls="--", lw=1.5, alpha=0.9)
    (points,) = axes[1].plot(*boundaries[0].T, "o", color="white", ms=3.2)
    error = axes[2].imshow(
        errors[0],
        cmap="inferno",
        vmin=0,
        vmax=float(errors.max()),
        interpolation="nearest",
    )
    axes[2].set_title(
        "Sampled-feature error", fontsize=22, color="#17314b", pad=15, weight="bold"
    )
    axes[2].set_xlabel("Dark: low error · bright: high error", fontsize=12, labelpad=12)
    fig.suptitle(
        "Fitting geometry through source visibility",
        fontsize=28,
        color="#17314b",
        weight="bold",
        y=0.965,
    )
    step_text = status.text(0, 0.67, "", fontsize=21, weight="bold", color="#17314b")
    loss_text = status.text(0, 0.30, "", fontsize=15, color="#52667a")
    (line,) = chart.plot(
        [], [], color="#147f9e", lw=2.8, label="With boundary gradients"
    )
    (control_line,) = chart.plot(
        [], [], color="#ca653e", lw=2, ls="--", label="Smooth gradients only"
    )
    chart.set(
        xlim=(0, steps),
        ylim=(-4, 108),
        xlabel="Optimization step",
        ylabel="Loss (% of initial)",
    )
    chart.spines[["top", "right"]].set_visible(False)
    chart.grid(alpha=0.18)
    chart.legend(frameon=False, loc="upper right", fontsize=11)
    for text in [chart.xaxis.label, chart.yaxis.label]:
        text.set_fontsize(12)

    def update(step):
        current.set_data(images[step])
        points.set_data(*boundaries[step].T)
        error.set_data(errors[step])
        line.set_data(np.arange(step + 1), losses[: step + 1])
        control_line.set_data(np.arange(step + 1), control_losses[: step + 1])
        step_text.set_text(f"Iteration {step} / {steps}")
        loss_text.set_text(f"Feature loss: {losses[step]:.1f}% of initial")
        fig.canvas.draw()

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
        "1600x900",
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
    for step in range(steps + 1):
        update(step)
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().tobytes()
        repeats = fps * 2 if step in (0, steps) else 1
        for _ in range(repeats):
            process.stdin.write(pixels)
        if step in (0, steps // 2, steps):
            fig.savefig(output / f"source_features_step_{step:03d}.png", dpi=100)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("video encoding failed")
    plt.close(fig)
    draw_comparison(output, scene, steps, boundaries, images, losses[-1])
    draw_samples(output, scene)


def draw_comparison(output, scene, steps, boundaries, images, final_loss_percent):
    target = scene.target.cpu().numpy()
    target_loop = np.concatenate((target, target[:1]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), dpi=110, facecolor="#f5f7fb")
    for ax, title, data in zip(
        axes,
        ["Target", "Initial mesh", f"After {steps} steps"],
        [as_image(scene.source), images[0], images[-1]],
        strict=True,
    ):
        ax.imshow(data, interpolation="nearest")
        ax.set_title(title, fontsize=23, color="#17314b", weight="bold", pad=16)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax, boundary in zip(axes[1:], [boundaries[0], boundaries[-1]], strict=True):
        ax.plot(*target_loop.T, color="white", ls="--", lw=1.5)
        ax.plot(*boundary.T, "o", color="white", ms=3.2)
    fig.suptitle(
        "Fitting an occluder from sampled features",
        fontsize=28,
        color="#17314b",
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.045,
        f"Feature loss falls by {100 - final_loss_percent:.1f}% · "
        "White dots: fitted vertices · Dashed line: target",
        ha="center",
        fontsize=15,
        color="#52667a",
    )
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.12, top=0.83, wspace=0.07)
    fig.savefig(output / "source_features_demo_poster.png", dpi=110)
    plt.close(fig)


def draw_samples(output, scene):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=110, facecolor="#f5f7fb")
    for ax, title, data in zip(
        axes,
        ["Source image", "Samples on the rear plane", "Visibility confidence"],
        [
            as_image(scene.source),
            as_image(scene.desired.reshape(1, 3, scene.size, scene.size)),
            scene.target_confidence.reshape(scene.size, scene.size).cpu().numpy(),
        ],
        strict=True,
    ):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=20, color="#17314b", pad=15)
        ax.set_axis_off()
    fig.text(
        0.5,
        0.035,
        "Queries behind the polygon have zero sampled features and zero confidence.",
        ha="center",
        fontsize=15,
        color="#52667a",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.86, wspace=0.08)
    fig.savefig(output / "source_features_sampling.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("source_features_demo"))
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.size < 16 or args.fps < 1:
        parser.error("steps and fps must be positive; size must be at least 16")
    args.output.mkdir(parents=True, exist_ok=True)
    scene = Scene(args.size, args.device)
    history, boundaries, images, errors = fit(scene, args.steps, edge_gradients=True)
    control, _, _, _ = fit(scene, args.steps, edge_gradients=False)
    draw_outputs(
        args.output, scene, history, control, boundaries, images, errors, args.fps
    )
    (args.output / "source_features_demo.json").write_text(
        json.dumps(
            {
                "size": args.size,
                "steps": args.steps,
                "edge_gradients": history,
                "smooth_only": control,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        args.output / "source_features_fit.npz",
        boundary=boundaries,
        target=scene.target.cpu().numpy(),
        feature_error=errors,
    )
    print(
        f"Feature MSE: {history[0]['feature_mse']:.6f} -> {history[-1]['feature_mse']:.6f}"
    )


if __name__ == "__main__":
    main()

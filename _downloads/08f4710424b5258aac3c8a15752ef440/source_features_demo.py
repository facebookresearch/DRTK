#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the root LICENSE file.
"""Fit translucent surfaces from shadows cast by three lights."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from drtk.source_features import sample_source_features, source_transmittance
from drtk.transparent_render import render_transparent


class Scene:
    def __init__(self, size=256, resolution=128, device="cuda"):
        self.size, self.resolution, self.device = size, resolution, device
        grid = 24
        self.segments = 24
        yy, xx = th.meshgrid(
            th.linspace(-3, 3, grid + 1, device=device),
            th.linspace(-3, 3, grid + 1, device=device),
            indexing="ij",
        )
        self.floor = th.stack((xx, yy, th.zeros_like(xx)), -1).reshape(-1, 3)
        faces = []
        for y in range(grid):
            for x in range(grid):
                a = y * (grid + 1) + x
                b = a + 1
                d = a + grid + 1
                c = d + 1
                faces.extend(((a, b, c), (a, c, d)))
        self.floor_faces = len(faces)
        for layer in range(2):
            offset = len(self.floor) + layer * (self.segments + 1)
            for i in range(self.segments):
                faces.append(
                    (offset, offset + 1 + i, offset + 1 + (i + 1) % self.segments)
                )
        self.faces = th.tensor(faces, device=device, dtype=th.int32)
        v, u = th.meshgrid(
            (th.arange(resolution, device=device) + 0.5) / resolution,
            (th.arange(resolution, device=device) + 0.5) / resolution,
            indexing="ij",
        )
        gx, gy = (u * 0.8 + 0.1) * grid, (v * 0.8 + 0.1) * grid
        ix, iy = gx.long(), gy.long()
        du, dv = gx - ix, gy - iy
        s = dv > du
        self.q_faces = ((iy * grid + ix) * 2 + s.int()).reshape(1, -1).int()
        self.q_bary = th.stack(
            (
                th.where(s, 1 - dv, 1 - du),
                th.where(s, du, du - dv),
                th.where(s, dv - du, dv),
            ),
            -1,
        ).reshape(1, -1, 3)
        self.theta = th.arange(self.segments, device=device) * 2 * th.pi / self.segments
        self.lights = th.tensor(
            [[-2, -1.5, 5.5], [2, -1.5, 5.5], [0.3, 2.5, 5.5]],
            device=device,
            dtype=th.float32,
        )
        forward = -self.lights
        forward = forward / forward.norm(dim=-1, keepdim=True)
        right = th.linalg.cross(
            forward, forward.new_tensor([0, 0, 1]).expand_as(forward)
        )
        right = right / right.norm(dim=-1, keepdim=True)
        up = th.linalg.cross(right, forward)
        self.rotation = th.stack((right, -up, forward), 1)
        self.target_centers = th.tensor(
            [[-0.35, -0.1, 0.8], [0.4, 0.25, 1.3]], device=device
        )
        self.initial_centers = th.tensor(
            [[-0.55, 0.15, 0.6], [0.6, -0.05, 1.05]], device=device
        )
        self.basis = th.stack(
            [th.ones_like(self.theta)]
            + [f(k * self.theta) for k in (2, 3, 4) for f in (th.cos, th.sin)]
        )
        self.target_coeffs = th.tensor(
            [
                [np.log(0.67), 0, 0.09, 0.15, -0.06, 0, 0],
                [np.log(0.57), 0.11, 0.17, 0.025, 0.12, 0, 0],
            ],
            device=device,
            dtype=th.float32,
        )
        self.initial_coeffs = th.tensor(
            [[np.log(0.53), 0, 0, 0, 0, 0, 0], [np.log(0.72), 0, 0, 0, 0, 0, 0]],
            device=device,
            dtype=th.float32,
        )
        self.target_radii = (self.target_coeffs @ self.basis).exp()
        self.initial_radii = th.stack(
            (th.full_like(self.theta, 0.53), th.full_like(self.theta, 0.72))
        )
        self.tilts = th.tensor([[0.14, -0.13], [-0.18, 0.08]], device=device)
        self.alpha_target = th.tensor([0.62, 0.78], device=device)

    def world(self, centers, radii):
        xy = (
            th.stack((th.cos(self.theta), 0.8 * th.sin(self.theta)), -1)[None]
            * radii[:, :, None]
        )
        z = (xy * self.tilts[:, None]).sum(-1, keepdim=True)
        ring = th.cat((xy, z), -1) + centers[:, None]
        panels = th.cat((centers[:, None], ring), 1).reshape(-1, 3)
        return th.cat((self.floor, panels))

    def project(self, world):
        camera = th.einsum(
            "kij,kvj->kvi", self.rotation, world[None] - self.lights[:, None]
        )
        return th.cat(
            (
                camera[..., :2] / camera[..., 2:] * self.size * 0.70
                + (self.size - 1) / 2,
                camera[..., 2:],
            ),
            -1,
        )

    def field(self, centers, radii, alpha, edge=True):
        world = self.world(centers, radii)
        opacity = th.cat(
            (
                world.new_ones(len(self.floor)),
                alpha[:, None].expand(-1, self.segments + 1).reshape(-1),
            )
        )
        _, w, _, _ = source_transmittance(
            self.project(world),
            self.faces,
            self.q_faces.expand(3, -1),
            self.q_bary.expand(3, -1, -1),
            self.size,
            self.size,
            vertex_opacity=opacity[None].expand(3, -1),
            edge_gradients=edge,
        )
        return w.sum(-1).reshape(3, self.resolution, self.resolution)


def fit(scene, steps, edge):
    with th.no_grad():
        target = scene.field(
            scene.target_centers, scene.target_radii, scene.alpha_target, False
        )
    centers = scene.initial_centers.clone().requires_grad_()
    logr = scene.initial_coeffs.clone().requires_grad_()
    logits = th.logit(th.tensor([0.4, 0.4], device=scene.device)).requires_grad_()
    opt = th.optim.Adam(
        [
            {"params": [centers], "lr": 0.012},
            {"params": [logr], "lr": 0.02},
            {"params": [logits], "lr": 0.035},
        ]
    )
    snapshots = []
    history = []
    t = time.monotonic()
    for step in range(steps + 1):
        opt.zero_grad()
        radii = (logr @ scene.basis).exp()
        alpha = logits.sigmoid()
        field = scene.field(centers, radii, alpha, edge)
        loss = (field - target).square().mean()
        if step % 4 == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "center_error": float(
                    (centers - scene.target_centers).square().mean().sqrt().detach()
                ),
                "radius_error": float(
                    (radii - scene.target_radii).square().mean().sqrt().detach()
                ),
                "opacity": alpha.detach().cpu().tolist(),
            }
            history.append(row)
            snapshots.append(
                {
                    "field": field.detach().cpu().numpy(),
                    "centers": centers.detach().cpu().numpy().copy(),
                    "radii": radii.detach().cpu().numpy().copy(),
                    "alpha": alpha.detach().cpu().numpy().copy(),
                }
            )
            if step % 100 == 0:
                print(edge, row, round(time.monotonic() - t, 2), flush=True)
        if step < steps:
            loss.backward()
            opt.step()
            if step + 1 in (steps // 2, 3 * steps // 4):
                for g in opt.param_groups:
                    g["lr"] *= 0.3
    return target.cpu().numpy(), history, snapshots


def check_transmittance(scene):
    with th.no_grad():
        centers, radii = scene.target_centers, scene.target_radii
        clear = scene.field(centers, radii, scene.alpha_target * 0, False)
        th.testing.assert_close(clear, th.ones_like(clear), rtol=0, atol=1e-5)
        target = scene.field(centers, radii, scene.alpha_target, False)
        levels = target.new_tensor([1, 0.38, 0.22, 0.0836])
        counts = ((target[..., None] - levels).abs() < 1e-5).sum((0, 1, 2))
        assert bool((counts > 10).all()), counts
        world = scene.world(centers, radii)
        opacity = th.cat(
            (
                world.new_ones(len(scene.floor)),
                scene.alpha_target[:, None].expand(-1, scene.segments + 1).reshape(-1),
            )
        )
        indices, weights, _, _ = source_transmittance(
            scene.project(world),
            scene.faces,
            scene.q_faces.expand(3, -1),
            scene.q_bary.expand(3, -1, -1),
            scene.size,
            scene.size,
            vertex_opacity=opacity[None].expand(3, -1),
        )
        lighting, confidence = sample_source_features(
            world.new_ones(3, 1, scene.size, scene.size), indices, weights
        )
        th.testing.assert_close(lighting, confidence)
        th.testing.assert_close(lighting.reshape_as(target), target)
    return {
        "transmittance_levels": levels.cpu().tolist(),
        "interior_sample_counts": counts.cpu().tolist(),
    }


def project_camera(world, position, look_at, size):
    eye = world.new_tensor(position)
    forward = world.new_tensor(look_at) - eye
    forward = forward / forward.norm()
    right = th.linalg.cross(forward, world.new_tensor([0, 0, 1]))
    right = right / right.norm()
    up = th.linalg.cross(right, forward)
    camera = (world - eye) @ th.stack((right, -up, forward)).T
    return th.cat(
        (camera[:, :2] / camera[:, 2:] * size * 1.8 + (size - 1) / 2, camera[:, 2:]), -1
    )


@th.no_grad()
def render_scene(scene, snapshot, size=400):
    centers = th.as_tensor(snapshot["centers"], device=scene.device)
    radii = th.as_tensor(snapshot["radii"], device=scene.device)
    alpha = th.as_tensor(snapshot["alpha"], device=scene.device)
    world = scene.world(centers, radii)
    field = th.as_tensor(snapshot["field"], device=scene.device)
    resolution = scene.resolution
    color = world.new_tensor([[0.12, 0.48, 0.80], [0.95, 0.42, 0.15]])
    texture = world.new_ones(3, 3, resolution, resolution + 32)
    floor_light = (0.12 + 0.88 * field)[:, None].expand(-1, 3, -1, -1)
    # Padding keeps the floor white when its UV lies outside the queried patch.
    texture[:, :, :, :resolution] = floor_light
    texture[:, :, :, resolution : resolution + 16] = color[0][None, :, None, None]
    texture[:, :, :, resolution + 16 :] = color[1][None, :, None, None]
    floor_uv = ((scene.floor[:, :2] + 2.4) / 4.8).clamp(0, 1)
    floor_uv = th.stack(
        ((floor_uv[:, 0] * (resolution - 1) + 0.5) / (resolution + 32), floor_uv[:, 1]),
        -1,
    )
    panel_uv = world.new_tensor(
        [
            [(resolution + 8) / (resolution + 32), 0.5],
            [(resolution + 24) / (resolution + 32), 0.5],
        ]
    )
    uv = th.cat(
        (floor_uv, panel_uv[:, None].expand(-1, scene.segments + 1, -1).reshape(-1, 2))
    )
    opacity = th.cat(
        (
            world.new_ones(len(scene.floor)),
            alpha[:, None].expand(-1, scene.segments + 1).reshape(-1),
        )
    )
    camera = project_camera(world, [4.2, -6.5, 6.2], [0, 0, 0.45], size)
    image = render_transparent(
        camera[None].expand(3, -1, -1),
        scene.faces,
        uv[None].expand(3, -1, -1),
        texture,
        size,
        size,
        opacity=opacity[None].expand(3, -1),
        background=world.new_tensor([0.95, 0.96, 0.98]),
        rgb_activation="identity",
        alpha_activation="clamp",
        edge_gradients=False,
    )[0]
    return (image.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def clean_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_explanation(output, scene, target):
    snapshot = {
        "centers": scene.target_centers.cpu().numpy(),
        "radii": scene.target_radii.cpu().numpy(),
        "alpha": scene.alpha_target.cpu().numpy(),
        "field": target,
    }
    image = render_scene(scene, snapshot, size=512)[2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=100, facecolor="#f5f7fb")
    axes[0].imshow(image)
    clean_axis(axes[0])
    axes[1].imshow(target[2], cmap="gray", vmin=0, vmax=1, origin="lower")
    clean_axis(axes[1])
    axes[0].set_title(
        "Translucent surfaces cast shadows", fontsize=16, color="#17314b", pad=16
    )
    axes[1].set_title(
        "Light reaching the receiving surface", fontsize=16, color="#17314b", pad=16
    )
    ax = axes[2]
    heights = [1, 0.38, 0.22, 0.0836]
    ax.bar(
        range(4),
        heights,
        color=["#c3cbd6", "#1f7abb", "#f26b26", "#484354"],
        width=0.65,
    )
    ax.set_xticks(range(4), ["Clear", "Blue", "Orange", "Both"], fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Transmittance", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    for i, h in enumerate(heights):
        ax.text(
            i, h + 0.045, f"{100 * h:g}%", ha="center", fontsize=13, color="#17314b"
        )
    ax.set_title(
        "Each layer attenuates the light", fontsize=16, color="#17314b", pad=16
    )
    fig.text(
        0.5,
        0.15,
        r"$T(q)=\prod_{f\prec q}(1-\alpha_f)$",
        ha="center",
        fontsize=21,
        color="#17314b",
    )
    fig.text(
        0.5,
        0.03,
        "Opacity: blue 0.62, orange 0.78. Where both overlap, 0.38 × 0.22 = 0.0836 of the light remains.",
        ha="center",
        fontsize=13,
        color="#52667a",
    )
    fig.subplots_adjust(left=0.025, right=0.98, bottom=0.29, top=0.89, wspace=0.18)
    fig.savefig(output / "source_features_sampling.png")
    plt.close(fig)


def make_figure(histories, snapshots, images, target):
    fig = plt.figure(figsize=(16, 11), dpi=100, facecolor="#f5f7fb")
    grid = fig.add_gridspec(
        3,
        3,
        left=0.035,
        right=0.975,
        top=0.845,
        bottom=0.09,
        height_ratios=[2.1, 1.25, 1.0],
        hspace=0.3,
        wspace=0.14,
    )
    renders = []
    fields = []
    for light in range(3):
        ax = fig.add_subplot(grid[0, light])
        renders.append(ax.imshow(images[0, light]))
        clean_axis(ax)
        ax.set_title(
            f"Light {light + 1}", fontsize=20, color="#17314b", weight="bold", pad=12
        )
        sub = grid[1, light].subgridspec(1, 2, wspace=0.08)
        for col, data in enumerate((target[light], snapshots[0]["field"][light])):
            ax = fig.add_subplot(sub[0, col])
            h = ax.imshow(data, vmin=0, vmax=1, cmap="gray", origin="lower")
            clean_axis(ax)
            ax.set_title(
                "Target shadow" if col == 0 else "Fitted shadow",
                fontsize=13,
                color="#52667a",
                pad=10,
            )
            if col == 1:
                fields.append(h)
    status = fig.add_subplot(grid[2, 0])
    status.axis("off")
    step_text = status.text(0.02, 0.88, "", fontsize=23, weight="bold", color="#17314b")
    alpha_text = status.text(
        0.02, 0.40, "", fontsize=15, color="#17314b", linespacing=1.5
    )
    status.text(0.02, 0.03, "Supervision: shadows only", fontsize=14, color="#52667a")
    chart = fig.add_subplot(grid[2, 1:])
    lines = []
    for color, label in [
        ("#127f9b", "With boundary gradients"),
        ("#c66b39", "Smooth gradients only"),
    ]:
        lines.append(chart.plot([], [], color=color, label=label, lw=2.4)[0])
    all_losses = [r["loss"] for rows in histories.values() for r in rows]
    chart.set(
        yscale="log",
        xlim=(0, histories["boundary"][-1]["step"]),
        ylim=(max(min(all_losses) * 0.5, 1e-17), max(all_losses) * 2),
        xlabel="Optimization step",
        ylabel="Shadow MSE",
    )
    chart.spines[["top", "right"]].set_visible(False)
    chart.grid(alpha=0.2)
    chart.legend(frameon=False, fontsize=12, loc="upper right")
    fig.suptitle(
        "Differentiable visibility",
        fontsize=29,
        color="#17314b",
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.93,
        "Fit shape, height and opacity from shadows cast by three lights.",
        ha="center",
        fontsize=18,
        color="#52667a",
    )
    fig.text(
        0.5,
        0.025,
        "The overlapping translucent surfaces produce darker partial shadows.",
        ha="center",
        fontsize=14,
        color="#52667a",
    )

    def update(index):
        row = histories["boundary"][index]
        for light in range(3):
            renders[light].set_data(images[index, light])
            fields[light].set_data(snapshots[index]["field"][light])
        for line, name in zip(lines, ("boundary", "smooth"), strict=True):
            rows = histories[name][: index + 1]
            line.set_data([r["step"] for r in rows], [r["loss"] for r in rows])
        step_text.set_text(f"Step {row['step']} / {histories['boundary'][-1]['step']}")
        a, b = row["opacity"]
        alpha_text.set_text(
            f"Blue opacity: {a:.3f}  (target 0.620)\nOrange opacity: {b:.3f}  (target 0.780)"
        )
        fig.canvas.draw()

    return fig, update


def write_video(output, histories, snapshots, images, target, fps):
    fig, update = make_figure(histories, snapshots, images, target)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1600x1100",
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
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for index in range(len(snapshots)):
        update(index)
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy().tobytes()
        repeat = 2 * fps if index in (0, len(snapshots) - 1) else 1
        for _ in range(repeat):
            process.stdin.write(pixels)
        if index in (0, len(snapshots) // 2, len(snapshots) - 1):
            fig.savefig(
                output
                / f"source_features_step_{histories['boundary'][index]['step']:03d}.png"
            )
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("video encoding failed")
    fig.savefig(output / "source_features_demo_poster.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("source_features_demo"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 4 or args.size < 32 or args.resolution < 16 or args.fps < 1:
        parser.error(
            "steps >= 4, size >= 32, resolution >= 16 and fps >= 1 are required"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    scene = Scene(args.size, args.resolution, args.device)
    checks = check_transmittance(scene)
    target, history, snapshots = fit(scene, args.steps, True)
    _, control, _ = fit(scene, args.steps, False)
    histories = {"boundary": history, "smooth": control}
    images = np.asarray([render_scene(scene, snapshot) for snapshot in snapshots])
    draw_explanation(args.output, scene, target)
    write_video(args.output, histories, snapshots, images, target, args.fps)
    np.savez_compressed(
        args.output / "source_features_fit.npz",
        target=target,
        **{key: np.asarray([s[key] for s in snapshots]) for key in snapshots[0]},
    )
    (args.output / "source_features_demo.json").write_text(
        json.dumps({"checks": checks, "histories": histories}, indent=2)
    )


if __name__ == "__main__":
    main()

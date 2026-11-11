# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fit mesh geometry and grouped topology decisions from rendered images."""

import argparse
import json
import math
import subprocess
import time
from functools import partial
from pathlib import Path

import torch
from drtk.pixel_grid_mesh import bernoulli_render_loss, pixel_grid_mesh
from drtk.transparent_render import render_transparent


class Scene:
    def __init__(self, resolution=96):
        self.height, self.width = 10, 14
        self.resolution = resolution
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.height, device="cuda"),
            torch.linspace(-1, 1, self.width, device="cuda"),
            indexing="ij",
        )
        self.depth = 1.2 + 0.14 * (x > 0).float()
        self.intrinsics = x.new_tensor([[14, 0, 6.5], [0, 14, 4.5], [0, 0, 1]])
        self.regions = [
            ((x + 0.5).abs() < 0.18) & (y.abs() < 0.25),
            ((x - 0.5).abs() < 0.18) & (y.abs() < 0.25),
            (x.abs() < 0.25) & (y < -0.65),
        ]
        self.basis = torch.stack(
            [
                torch.ones_like(x),
                x,
                y,
                torch.sin(math.pi * x),
                torch.sin(math.pi * y),
                torch.sign(x),
            ],
            dim=-1,
        )
        self.target_coefficients = x.new_tensor(
            [
                [0.0, 0.01, 0.0],
                [0.015, 0.025, 0.025],
                [0.01, 0.0, 0.02],
                [0.0, 0.02, 0.095],
                [0.02, 0.015, 0.04],
                [0.055, 0.0, 0.0],
            ]
        )
        self.target_bits = torch.tensor(
            [False, False, True, False, True], device="cuda"
        )
        self.diagonal = torch.zeros_like(x, dtype=torch.bool)
        self.feather = torch.full_like(x, 0.08)
        self.texture = self.make_texture()
        self.views = [(-25.0, -8.0), (0.0, 0.0), (25.0, 8.0)]
        with torch.no_grad():
            self.targets = [
                self.render(self.target_bits[None], self.target_coefficients, view)[:2]
                for view in self.views
            ]

    def make_texture(self):
        y, x = torch.meshgrid(
            torch.linspace(0, 1, 160, device="cuda"),
            torch.linspace(0, 1, 224, device="cuda"),
            indexing="ij",
        )
        left = x.new_tensor([0.10, 0.66, 0.83])
        right = x.new_tensor([0.94, 0.38, 0.16])
        mix = (5 * (x - 0.4)).clamp(0, 1)[..., None]
        rgb = left * (1 - mix) + right * mix
        wave = 0.15 * torch.sin(5 * x + 3 * y)[..., None]
        rgb = (rgb + wave).clamp(0, 1)
        grid = ((14 * x).remainder(1) < 0.04) | ((10 * y).remainder(1) < 0.04)
        dots = (
            ((7 * x).remainder(1) - 0.5).square()
            + ((5 * y).remainder(1) - 0.5).square()
        ) < 0.018
        rgb = torch.where(grid[..., None], rgb * 0.55, rgb)
        rgb = torch.where(dots[..., None], torch.ones_like(rgb) * 0.95, rgb)
        return rgb.permute(2, 0, 1)[None].contiguous()

    def topology(self, bits):
        batch = bits.shape[0]
        alive = torch.ones(
            batch, self.height, self.width, device="cuda", dtype=torch.bool
        )
        for index, region in enumerate(self.regions):
            alive = alive & (~region | bits[:, index, None, None])
        right = torch.ones(
            batch, self.height, self.width - 1, device="cuda", dtype=torch.bool
        )
        down = torch.ones(
            batch, self.height - 1, self.width, device="cuda", dtype=torch.bool
        )
        right[:, :, self.width // 2 - 1] = bits[:, 3, None]
        down[:, self.height // 2 - 1, :] = bits[:, 4, None]
        return alive, right, down

    def mesh(self, bits, coefficients):
        batch = bits.shape[0]
        return pixel_grid_mesh(
            self.depth.expand(batch, -1, -1),
            (self.basis @ coefficients).expand(batch, -1, -1, -1),
            self.intrinsics,
            *self.topology(bits),
            diagonal=self.diagonal.expand(batch, -1, -1),
            feather_width=self.feather.expand(batch, -1, -1),
        )

    def project(self, vertices, view, resolution, framing=1.0):
        yaw, pitch = map(math.radians, view)
        ry = vertices.new_tensor(
            [
                [math.cos(yaw), 0, math.sin(yaw)],
                [0, 1, 0],
                [-math.sin(yaw), 0, math.cos(yaw)],
            ]
        )
        rx = vertices.new_tensor(
            [
                [1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)],
            ]
        )
        center = vertices.new_tensor([0, 0, 1.27])
        camera = (vertices - center) @ (rx @ ry).T + center
        screen = camera[..., :2] / camera[..., 2:] * (resolution * 0.78 * framing)
        screen = screen + (resolution - 1) / 2
        return torch.cat((screen, camera[..., 2:]), dim=-1)

    def render(self, bits, coefficients, view, resolution=None, framing=1.0):
        size = resolution or self.resolution
        v, uv, _, vi, opacity, _, _ = self.mesh(bits, coefficients)
        rgb, alpha, _ = render_transparent(
            self.project(v, view, size, framing),
            vi,
            uv,
            self.texture,
            size,
            size,
            opacity=opacity,
            rgb_activation="identity",
            alpha_activation="clamp",
        )
        return rgb, alpha, (v, vi)

    def loss(self, bits, coefficients, view_index):
        rgb, alpha, _ = self.render(bits, coefficients, self.views[view_index])
        target_rgb, target_alpha = self.targets[view_index]
        return (rgb - target_rgb).square().mean((1, 2, 3)) + 0.25 * (
            alpha - target_alpha
        ).square().mean((1, 2, 3))


def optimize(scene, steps, pairs, output):
    logits = torch.nn.Parameter(torch.full((5,), 1.1, device="cuda"))
    coefficients = torch.nn.Parameter(torch.zeros(6, 3, device="cuda"))
    optimizer = torch.optim.Adam(
        [
            {"params": [coefficients], "lr": 0.002},
            {"params": [logits], "lr": 0.03},
        ]
    )
    generator = torch.Generator(device="cuda").manual_seed(2026)
    states, history = [], []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                loss = sum(
                    scene.loss((logits > 0)[None], coefficients, i).mean()
                    for i in range(len(scene.views))
                ) / len(scene.views)
                row = {
                    "step": step,
                    "loss": float(loss),
                    "probabilities": logits.sigmoid().tolist(),
                    "seconds": time.perf_counter() - started,
                }
                history.append(row)
                states.append(
                    {
                        "step": step,
                        "logits": logits.detach().cpu().clone(),
                        "coefficients": coefficients.detach().cpu().clone(),
                    }
                )
                if step % 100 == 0 or step == steps:
                    print(json.dumps(row), flush=True)
        if step == steps:
            break
        view_index = step % len(scene.views)

        evaluate = partial(scene.loss, coefficients=coefficients, view_index=view_index)

        optimizer.zero_grad()
        loss = bernoulli_render_loss(logits, evaluate, pairs=pairs, generator=generator)
        loss.backward()
        optimizer.step()
    torch.save(
        {"states": states, "history": history, "steps": steps, "pairs": pairs},
        output / "checkpoint.pt",
    )
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return states, history


def render_media(scene, states, history, output):  # noqa: C901
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont

    background = (15, 23, 35)
    foreground = (235, 241, 249)
    muted = (160, 177, 196)
    teal = (65, 211, 197)
    orange = (255, 173, 88)
    names = [
        "Left patch alive",
        "Right patch alive",
        "Top patch alive",
        "Vertical seam joined",
        "Horizontal seam joined",
    ]

    def font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)

    @torch.no_grad()
    def rendered(bits, coefficients, view, size):
        rgb, alpha, _ = scene.render(bits[None], coefficients, view, size, framing=0.8)
        color = (
            rgb
            + (1 - alpha) * rgb.new_tensor([0.045, 0.065, 0.09])[None, :, None, None]
        )
        pixels = (255 * color[0].permute(1, 2, 0).clamp(0, 1)).byte()
        return Image.fromarray(pixels.cpu().numpy())

    def topology(bits, size=(400, 285)):
        alive, right, down = [t[0].cpu().numpy() for t in scene.topology(bits[None])]
        image = Image.new("RGB", size, background)
        draw = ImageDraw.Draw(image)
        sx, sy = size[0] / scene.width, size[1] / scene.height
        for y in range(scene.height):
            for x in range(scene.width):
                if alive[y, x]:
                    draw.rectangle(
                        (x * sx + 1, y * sy + 1, (x + 1) * sx - 1, (y + 1) * sy - 1),
                        fill=teal,
                    )
                if x < scene.width - 1 and alive[y, x] and alive[y, x + 1]:
                    if not right[y, x]:
                        draw.line(
                            ((x + 1) * sx, y * sy, (x + 1) * sx, (y + 1) * sy),
                            fill=orange,
                            width=4,
                        )
                if y < scene.height - 1 and alive[y, x] and alive[y + 1, x]:
                    if not down[y, x]:
                        draw.line(
                            (x * sx, (y + 1) * sy, (x + 1) * sx, (y + 1) * sy),
                            fill=orange,
                            width=4,
                        )
        return image

    initial = states[0]
    final = states[-1]
    variants = [
        ("Initial", initial["logits"].cuda() > 0, initial["coefficients"].cuda()),
        ("Target", scene.target_bits, scene.target_coefficients),
        ("Optimized", final["logits"].cuda() > 0, final["coefficients"].cuda()),
    ]
    result = Image.new("RGB", (1320, 930), background)
    draw = ImageDraw.Draw(result)
    for col, (name, bits, coefficients) in enumerate(variants):
        x = 15 + col * 440
        draw.text((x + 12, 15), name, fill=foreground, font=font(28))
        for row, view in enumerate([(0.0, 0.0), (32.0, -12.0)]):
            result.paste(rendered(bits, coefficients, view, 420), (x, 60 + row * 430))
    result.save(output / "result.png")
    masks = Image.new("RGB", (1320, 375), background)
    draw = ImageDraw.Draw(masks)
    for col, (name, bits, _) in enumerate(variants):
        x = 15 + col * 440
        draw.text((x + 12, 12), name, fill=foreground, font=font(26))
        masks.paste(topology(bits), (x + 12, 58))
    draw.text(
        (27, 348),
        "Teal: live cells    Orange: separated neighbors",
        fill=muted,
        font=font(17),
    )
    masks.save(output / "topology.png")

    steps = [r["step"] for r in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].semilogy(steps, [r["loss"] for r in history], color="#008b88")
    axes[0].set(
        xlabel="Iteration",
        ylabel="RGB + alpha loss",
        title="Thresholded mesh, three views",
    )
    for index, name in enumerate(names):
        axes[1].plot(steps, [r["probabilities"][index] for r in history], label=name)
    axes[1].set(
        xlabel="Iteration",
        ylabel="Probability",
        ylim=(-0.02, 1.02),
        title="Learned topology decisions",
    )
    axes[1].legend(fontsize=8, loc="center right")
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.savefig(output / "learning.png", dpi=160)
    plt.close(figure)

    def frame(state_index, view=(0.0, 0.0), orbit=False):
        state = states[state_index]
        row = history[state_index]
        bits = state["logits"].cuda() > 0
        image = Image.new("RGB", (1280, 720), background)
        draw = ImageDraw.Draw(image)
        draw.text(
            (24, 18), "Learning geometry and topology", fill=foreground, font=font(30)
        )
        draw.text((24, 76), "Target", fill=muted, font=font(22))
        draw.text(
            (464, 76),
            "Optimized mesh" if orbit else "Current mesh",
            fill=teal,
            font=font(22),
        )
        image.paste(
            rendered(scene.target_bits, scene.target_coefficients, view, 420), (24, 114)
        )
        image.paste(rendered(bits, state["coefficients"].cuda(), view, 420), (464, 114))
        draw.text((920, 80), "Hard topology", fill=foreground, font=font(21))
        image.paste(topology(bits, (330, 235)), (920, 119))
        draw.text((920, 368), "Learned probabilities", fill=foreground, font=font(20))
        for i, name in enumerate(names):
            probability = row["probabilities"][i]
            y = 408 + i * 49
            draw.text((920, y), name, fill=muted, font=font(15))
            draw.rectangle((920, y + 23, 1190, y + 33), fill=(40, 56, 74))
            draw.rectangle((920, y + 23, 920 + 270 * probability, y + 33), fill=teal)
            draw.text((1200, y + 16), f"{probability:.2f}", fill=muted, font=font(15))
        draw.text(
            (24, 572),
            f"Iteration {row['step']} / {history[-1]['step']}",
            fill=foreground,
            font=font(25),
        )
        draw.text(
            (24, 614),
            f"Three-view RGB + alpha loss: {row['loss']:.6f}",
            fill=muted,
            font=font(20),
        )
        draw.text(
            (24, 671),
            "Final mesh, changing viewpoint"
            if orbit
            else "Both geometry and topology train from the first iteration",
            fill=muted,
            font=font(18),
        )
        return image

    frame(len(states) - 1).save(output / "poster.png")
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
        "1280x720",
        "-r",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-threads",
        "2",
        "-vf",
        "scale=in_range=full:out_range=tv:out_color_matrix=bt709",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        str(output / "optimization.mp4"),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    last_key, pixels = None, None
    for i in range(600):
        orbit = i >= 450
        index = min(len(states) - 1, max(0, round((i - 30) / 419 * (len(states) - 1))))
        phase = (i - 450) / 149 if orbit else 0
        view = (
            (32 * math.sin(phase * 2 * math.pi), -12 * math.sin(phase * 2 * math.pi))
            if orbit
            else (0.0, 0.0)
        )
        key = (index, view, orbit)
        if key != last_key:
            pixels = frame(index, view, orbit).tobytes()
            last_key = key
        process.stdin.write(pixels)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("Video encoding failed")
    summary = {
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "relative_loss_reduction": 1 - history[-1]["loss"] / history[0]["loss"],
        "final_bits": (final["logits"] > 0).tolist(),
        "target_bits": scene.target_bits.tolist(),
        "recorded_states": len(states),
        "video_seconds": 20,
        "interpolated_parameters": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("pixel_grid_demo"))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    scene = Scene()
    if args.render_only:
        saved = torch.load(args.output / "checkpoint.pt", weights_only=True)
        states, history = saved["states"], saved["history"]
    else:
        states, history = optimize(scene, args.steps, args.pairs, args.output)
    render_media(scene, states, history, args.output)


if __name__ == "__main__":
    main()

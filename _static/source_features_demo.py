#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the root LICENSE file.
"""Fit opaque and translucent surfaces from shadows cast by three lights."""

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
import torch.nn.functional as F
from drtk.source_features import sample_with_transmittance, source_transmittance
from drtk.transparent_render import render_transparent


def receiver_queries(resolution, extent, device):
    grid = 48
    v, u = th.meshgrid(
        (th.arange(resolution, device=device) + 0.5) / resolution,
        (th.arange(resolution, device=device) + 0.5) / resolution,
        indexing="ij",
    )
    gx = (u * (extent / 2.8) + 0.5 * (1 - extent / 2.8)) * grid
    gy = (v * (extent / 2.8) + 0.5 * (1 - extent / 2.8)) * grid
    ix, iy = gx.long(), gy.long()
    du, dv = gx - ix, gy - iy
    upper = dv > du
    faces = ((iy * grid + ix) * 2 + upper.int()).reshape(1, -1).int()
    bary = th.stack(
        (
            th.where(upper, 1 - dv, 1 - du),
            th.where(upper, du, du - dv),
            th.where(upper, dv - du, dv),
        ),
        -1,
    ).reshape(1, -1, 3)
    return faces, bary


class Scene:
    def __init__(self, size=1024, resolution=512, device="cuda"):
        self.size, self.resolution, self.device = size, resolution, device
        self.object_count = 3
        self.object_scale = th.tensor([1.5, 1.0, 1.0], device=device)
        grid = 48
        self.segments = 32
        self.rings = 8
        self.panel_vertices = 1 + self.segments * self.rings
        yy, xx = th.meshgrid(
            th.linspace(-2.8, 2.8, grid + 1, device=device),
            th.linspace(-2.8, 2.8, grid + 1, device=device),
            indexing="ij",
        )
        distance = th.maximum((xx + 0.1).abs(), (yy - 0.1).abs())
        ramp = ((distance - 0.8) / 0.75).clamp(0, 1)
        height = 0.55 * ramp.square() * (3 - 2 * ramp) - 0.7
        self.floor = th.stack((xx, yy, height), -1).reshape(-1, 3)
        self.floor[:, :2] *= 1.5
        faces = []
        for y in range(grid):
            for x in range(grid):
                a = y * (grid + 1) + x
                b = a + 1
                d = a + grid + 1
                c = d + 1
                faces.extend(((a, b, c), (a, c, d)))
        self.floor_faces = len(faces)
        panel_faces = [
            (0, 1 + i, 1 + (i + 1) % self.segments) for i in range(self.segments)
        ]
        for ring in range(self.rings - 1):
            for i in range(self.segments):
                a = 1 + ring * self.segments + i
                b = a + self.segments
                c = 1 + (ring + 1) * self.segments + (i + 1) % self.segments
                d = c - self.segments
                panel_faces.extend(((a, b, c), (a, c, d)))
        self.panel_faces = th.tensor(panel_faces, device=device, dtype=th.int32)
        for layer in range(self.object_count):
            offset = len(self.floor) + layer * self.panel_vertices
            faces.extend(tuple(offset + i for i in tri) for tri in panel_faces)
        self.faces = th.tensor(faces, device=device, dtype=th.int32)
        self.q_faces, self.q_bary = receiver_queries(resolution, 2.6, device)
        self.display_faces, self.display_bary = receiver_queries(
            resolution, 2.8, device
        )
        self.theta = th.arange(self.segments, device=device) * 2 * th.pi / self.segments
        self.lights = th.tensor(
            [[-1.6, -1.1, 6.85], [1.7, -1.1, 6.85], [0.3, 1.9, 6.85]],
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
            [[-0.6, -0.25, 1.6], [0.4, -0.15, 2.9], [0.25, -0.1, 3.5]], device=device
        )
        self.initial_centers = th.tensor(
            [[-0.8, -0.05, 1.4], [0.6, -0.4, 2.7], [0.05, -0.25, 3.65]], device=device
        )
        self.basis = th.stack(
            [th.ones_like(self.theta)]
            + [f(k * self.theta) for k in (2, 3, 4) for f in (th.cos, th.sin)]
        )
        self.target_coeffs = th.tensor(
            [
                [np.log(1.0), 0, 0.09, 0.15, -0.06, 0, 0],
                [np.log(1.0), 0.11, 0.17, 0.025, 0.12, 0, 0],
                [np.log(1.0), -0.12, 0.06, 0.1, -0.08, 0.05, 0],
            ],
            device=device,
            dtype=th.float32,
        )
        self.initial_coeffs = th.tensor(
            [
                [np.log(0.65), 0.25, -0.20, -0.14, 0.18, 0.06, -0.06],
                [np.log(1.25), -0.25, 0.08, -0.08, -0.16, 0.08, 0.02],
                [np.log(0.8), 0.18, -0.16, -0.15, 0.12, -0.06, 0.04],
            ],
            device=device,
            dtype=th.float32,
        )
        self.target_radii = (self.target_coeffs @ self.basis).exp()
        self.initial_radii = (self.initial_coeffs @ self.basis).exp()
        self.tilts = th.tensor(
            [[0.14, -0.13], [-0.18, 0.08], [0.08, 0.12]], device=device
        )
        self.bowing = th.tensor([0.24, 0.30, 0.27], device=device)
        self.alpha_target = th.tensor([1.0, 0.8, 0.3], device=device)
        self.alpha_initial = th.tensor([0.3, 0.5, 1.0], device=device)

    def world(self, centers, radii):
        fraction = th.arange(1, self.rings + 1, device=self.device) / self.rings
        direction = th.stack((self.theta.cos(), 0.8 * self.theta.sin()), -1)
        xy = (
            radii[:, None, :, None]
            * fraction[None, :, None, None]
            * direction[None, None]
        )
        bow = self.bowing[:, None, None] * (1 - fraction.square())[None, :, None]
        wave = (
            0.055
            * fraction.square()[None, :, None]
            * th.sin(3 * self.theta + 0.4)[None, None]
        )
        z = (xy * self.tilts[:, None, None]).sum(-1) + bow + wave
        rings = th.cat((xy, z[..., None]), -1).reshape(self.object_count, -1, 3)
        peak = centers.new_zeros(self.object_count, 1, 3)
        peak[:, :, 2] = self.bowing[:, None]
        panels = (
            th.cat((peak, rings), 1) * self.object_scale[:, None, None]
            + centers[:, None]
        )
        return th.cat((self.floor, panels.reshape(-1, 3)))

    def project(self, world):
        camera = th.einsum(
            "kij,kvj->kvi", self.rotation, world[None] - self.lights[:, None]
        )
        return th.cat(
            (
                camera[..., :2] / camera[..., 2:] * self.size * 0.35
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
                alpha[:, None].expand(-1, self.panel_vertices).reshape(-1),
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
    alpha = scene.alpha_initial.clone().requires_grad_()
    opt = th.optim.Adam(
        [
            {"params": [centers], "lr": 0.012},
            {"params": [logr], "lr": 0.02},
            {"params": [alpha], "lr": 0.02},
        ]
    )
    snapshots = []
    history = []
    t = time.monotonic()
    for step in range(steps + 1):
        opt.zero_grad()
        radii = (logr @ scene.basis).exp()
        field = scene.field(centers, radii, alpha, edge)
        loss = (field - target).square().mean()
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "center_error": float(
                (centers - scene.target_centers).square().mean().sqrt().detach()
            ),
            "radius_error": float(
                ((radii - scene.target_radii) * scene.object_scale[:, None])
                .square()
                .mean()
                .sqrt()
                .detach()
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
            with th.no_grad():
                alpha.clamp_(0, 1)
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
        levels = target.new_tensor([1, 0, 0.2, 0.7, 0.14])
        counts = ((target[..., None] - levels).abs() < 1e-5).sum((0, 1, 2))
        assert bool((counts > 10).all()), counts
        world = scene.world(centers, radii)
        opacity = th.cat(
            (
                world.new_ones(len(scene.floor)),
                scene.alpha_target[:, None]
                .expand(-1, scene.panel_vertices)
                .reshape(-1),
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
        lighting, confidence = sample_with_transmittance(
            world.new_ones(3, 1, scene.size, scene.size), indices, weights
        )
        th.testing.assert_close(lighting, confidence)
        th.testing.assert_close(lighting.reshape_as(target), target)
    return {
        "transmittance_levels": levels.cpu().tolist(),
        "interior_sample_counts": counts.cpu().tolist(),
    }


def project_camera(world, position, look_at, width, height):
    eye = world.new_tensor(position)
    forward = world.new_tensor(look_at) - eye
    forward = forward / forward.norm()
    right = th.linalg.cross(forward, world.new_tensor([0, 0, 1]))
    right = right / right.norm()
    up = th.linalg.cross(right, forward)
    camera = (world - eye) @ th.stack((right, -up, forward)).T
    return th.cat(
        (
            camera[:, :2] / camera[:, 2:] * width * 1.05
            + world.new_tensor([(width - 1) / 2, (width * 5 / 4 - 1) / 2]),
            camera[:, 2:],
        ),
        -1,
    )


def receiver_samples(scene, world, face_ids, barycentrics):
    triangles = world[scene.faces[face_ids.clamp_min(0).long()].long()]
    points = (triangles * barycentrics[..., None]).sum(-2)
    mesh_triangles = world[scene.faces.long()]
    face_normals = th.linalg.cross(
        mesh_triangles[:, 1] - mesh_triangles[:, 0],
        mesh_triangles[:, 2] - mesh_triangles[:, 0],
    )
    vertex_normals = th.zeros_like(world)
    for corner in range(3):
        vertex_normals.index_add_(0, scene.faces[:, corner].long(), face_normals)
    vertex_normals = F.normalize(vertex_normals, dim=-1)
    normals = F.normalize(
        (
            vertex_normals[scene.faces[face_ids.clamp_min(0).long()].long()]
            * barycentrics[..., None]
        ).sum(-2),
        dim=-1,
    )
    return points, normals


def diffuse(scene, points, normals, transmittance, color):
    direction = scene.lights[:, None] - points[None]
    distance2 = direction.square().sum(-1)
    cosine = (F.normalize(direction, dim=-1) * normals[None]).sum(-1).clamp_min(0)
    irradiance = 0.055 + 16.0 * cosine * transmittance / distance2
    return irradiance[..., None] * color


def panel_uv(scene):
    fraction = th.arange(1, scene.rings + 1, device=scene.device) / scene.rings
    direction = th.stack((scene.theta.cos(), scene.theta.sin()), -1)
    rings = (fraction[:, None, None] * direction[None]).reshape(-1, 2)
    return th.cat((rings.new_zeros(1, 2), rings)) * 0.45 + 0.5


def panel_queries(scene, resolution=128):
    if hasattr(scene, "_panel_queries"):
        return scene._panel_queries
    v, u = th.meshgrid(
        (th.arange(resolution, device=scene.device) + 0.5) / resolution,
        (th.arange(resolution, device=scene.device) + 0.5) / resolution,
        indexing="ij",
    )
    points = th.stack((u, v), -1).reshape(-1, 2)
    triangles = panel_uv(scene)[scene.panel_faces.long()]
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    delta = points[None] - triangles[:, 0, None]
    determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    b1 = (
        delta[..., 0] * e2[:, None, 1] - delta[..., 1] * e2[:, None, 0]
    ) / determinant[:, None]
    b2 = (
        delta[..., 1] * e1[:, None, 0] - delta[..., 0] * e1[:, None, 1]
    ) / determinant[:, None]
    all_bary = th.stack((1 - b1 - b2, b1, b2), -1)
    cover = (all_bary >= -1e-6).all(-1)
    selected = cover.int().argmax(0)
    valid = cover.any(0)
    bary = all_bary[selected, th.arange(len(points), device=scene.device)].clamp_min(0)
    bary = bary / bary.sum(-1, keepdim=True)
    ids = th.cat(
        [
            th.where(
                valid, scene.floor_faces + k * len(scene.panel_faces) + selected, -1
            )
            for k in range(scene.object_count)
        ]
    )
    scene._panel_queries = (
        ids[None].int(),
        bary.repeat(scene.object_count, 1)[None],
        valid,
    )
    return scene._panel_queries


def sphere_mesh(center, radius=0.045, rings=10, segments=20):
    latitude = th.linspace(0, th.pi, rings + 1, device=center.device)
    longitude = th.arange(segments, device=center.device) * 2 * th.pi / segments
    v, u = th.meshgrid(latitude, longitude, indexing="ij")
    xyz = th.stack((v.sin() * u.cos(), v.sin() * u.sin(), v.cos()), -1).reshape(-1, 3)
    faces = []
    for j in range(rings):
        for i in range(segments):
            a = j * segments + i
            b = j * segments + (i + 1) % segments
            c = b + segments
            d = a + segments
            faces.extend(((a, b, c), (a, c, d)))
    return center + radius * xyz, th.tensor(faces, device=center.device, dtype=th.int32)


def platform_sides(world, segments=48):
    vertices = []
    faces = []
    corners = [[-4.2, -4.2], [4.2, -4.2], [4.2, 4.2], [-4.2, 4.2]]
    for side in range(4):
        start, end = np.asarray(corners[side]), np.asarray(corners[(side + 1) % 4])
        for i in range(segments):
            a = start + (end - start) * i / segments
            b = start + (end - start) * (i + 1) / segments
            offset = len(vertices)
            vertices.extend([[*a, -0.15], [*a, -0.88], [*b, -0.88], [*b, -0.15]])
            faces.extend(
                ((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3))
            )
    return world.new_tensor(vertices), th.tensor(
        faces, device=world.device, dtype=th.int32
    )


@th.no_grad()
def render_scene(scene, snapshot, size=768):
    height = size * 4 // 3
    centers = th.as_tensor(snapshot["centers"], device=scene.device)
    radii = th.as_tensor(snapshot["radii"], device=scene.device)
    alpha = th.as_tensor(snapshot["alpha"], device=scene.device)
    world = scene.world(centers, radii)
    opacity = th.cat(
        (
            world.new_ones(len(scene.floor)),
            alpha[:, None].expand(-1, scene.panel_vertices).reshape(-1),
        )
    )
    r = scene.resolution
    objects = scene.object_count
    tile = r + 2
    atlas_end = (objects + 1) * tile
    # Display lighting covers the full receiver, including its outer rim.
    # At the physical boundary, condition the filter on taps on the surface;
    # silhouette coverage is applied by the final rasterization.
    projected_world = scene.project(world)
    if not hasattr(scene, "_display_coverage"):
        _, clear_weights, _, _ = source_transmittance(
            projected_world,
            scene.faces,
            scene.display_faces.expand(3, -1),
            scene.display_bary.expand(3, -1, -1),
            scene.size,
            scene.size,
            vertex_opacity=world.new_zeros(3, len(world)),
            edge_gradients=False,
        )
        scene._display_coverage = clear_weights.sum(-1)
    _, receiver_weights, _, _ = source_transmittance(
        projected_world,
        scene.faces,
        scene.display_faces.expand(3, -1),
        scene.display_bary.expand(3, -1, -1),
        scene.size,
        scene.size,
        vertex_opacity=opacity[None].expand(3, -1),
        edge_gradients=False,
    )
    field = receiver_weights.sum(-1) / scene._display_coverage.clamp_min(1e-8)
    points, normals = receiver_samples(
        scene, world, scene.display_faces[0], scene.display_bary[0]
    )
    checker = (
        (th.floor(points[:, 0] / 0.45) + th.floor(points[:, 1] / 0.45)).long() % 2
    ).float()
    material = (0.4525 + 0.05 * checker[:, None]) * world.new_tensor([0.91, 0.96, 1.0])[
        None
    ]
    floor_rgb = (
        diffuse(scene, points, normals, field, material)
        .reshape(3, r, r, 3)
        .permute(0, 3, 1, 2)
    )
    qf, qb, valid = panel_queries(scene)
    _, weights, _, _ = source_transmittance(
        scene.project(world),
        scene.faces,
        qf.expand(3, -1),
        qb.expand(3, -1, -1),
        scene.size,
        scene.size,
        vertex_opacity=opacity[None].expand(3, -1),
        edge_gradients=False,
    )
    transmitted = weights.sum(-1)
    transmitted = th.where(
        valid.repeat(objects)[None], transmitted, th.ones_like(transmitted)
    )
    panel_points, panel_normals = receiver_samples(scene, world, qf[0], qb[0])
    colors = world.new_tensor(
        [[0.025, 0.24, 0.80], [0.90, 0.17, 0.025], [0.03, 0.85, 0.28]]
    )
    rgb = diffuse(
        scene,
        panel_points,
        panel_normals,
        transmitted,
        colors[:, None].expand(-1, 128 * 128, -1).reshape(-1, 3),
    )
    panels = rgb.reshape(3, objects, 128, 128, 3).permute(0, 1, 4, 2, 3)
    panels = F.interpolate(
        panels.reshape(3 * objects, 3, 128, 128),
        size=(r, r),
        mode="bilinear",
        align_corners=False,
    ).reshape(3, objects, 3, r, r)
    sides, side_faces = platform_sides(world)
    side_points = sides.reshape(4, -1, 4, 3).mean((1, 2))
    side_tri = sides[side_faces[:: (len(side_faces) // 4)].long()]
    side_normals = F.normalize(
        th.linalg.cross(
            side_tri[:, 1] - side_tri[:, 0], side_tri[:, 2] - side_tri[:, 0]
        ),
        dim=-1,
    )
    side_rgb = diffuse(
        scene,
        side_points,
        side_normals,
        world.new_ones(3, 4),
        world.new_tensor([0.28, 0.31, 0.36]),
    )
    width = atlas_end + 5 * 8
    texture = world.new_zeros(3, 3, tile, width)
    texture[..., :tile] = F.pad(floor_rgb, (1, 1, 1, 1), mode="replicate")
    for obj in range(objects):
        texture[..., (obj + 1) * tile : (obj + 2) * tile] = F.pad(
            panels[:, obj], (1, 1, 1, 1), mode="replicate"
        )
    for side in range(4):
        texture[..., atlas_end + side * 8 : atlas_end + (side + 1) * 8] = side_rgb[
            :, side, :, None, None
        ]
    texture[..., atlas_end + 32 :] = 16
    floor_uv = (scene.floor[:, :2] / world.new_tensor([1.5, 1.5]) + 2.8) / 5.6
    floor_uv = th.stack(
        ((floor_uv[:, 0] * r + 1) / width, (floor_uv[:, 1] * r + 1) / tile), -1
    )
    disk = panel_uv(scene)
    uv = th.cat(
        [floor_uv]
        + [
            th.stack(
                (
                    (disk[:, 0] * r + (i + 1) * tile + 1) / width,
                    (disk[:, 1] * r + 1) / tile,
                ),
                -1,
            )
            for i in range(objects)
        ]
    )
    side_uv = th.stack(
        (
            (atlas_end + th.arange(4, device=scene.device) * 8 + 4) / width,
            world.new_full((4,), 0.5),
        ),
        -1,
    ).repeat_interleave(len(sides) // 4, 0)
    geometry = []
    for light in scene.lights:
        emitter, emitter_faces = sphere_mesh(light)
        geometry.append(th.cat((world, sides, emitter)))
    faces = th.cat(
        (scene.faces, side_faces + len(world), emitter_faces + len(world) + len(sides))
    )
    uv = th.cat(
        (
            uv,
            side_uv,
            world.new_tensor([[(atlas_end + 36) / width, 0.5]]).expand(
                len(emitter), -1
            ),
        )
    )
    opacity = th.cat(
        (opacity, world.new_ones(len(sides)), world.new_ones(len(emitter)))
    )
    camera = th.stack(
        [
            project_camera(v, [4.5, -7.0, 7.0], [0, 0, 2.2], size, height)
            for v in geometry
        ]
    )
    image = render_transparent(
        camera,
        faces,
        uv[None].expand(3, -1, -1),
        texture,
        height,
        size,
        opacity=opacity[None].expand(3, -1),
        background=world.new_zeros(3),
        rgb_activation="identity",
        alpha_activation="clamp",
        edge_gradients=False,
    )[0]
    image = th.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * image.clamp_min(0).pow(1 / 2.4) - 0.055,
    ).clamp(0, 1)
    return (image * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


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
    image = render_scene(scene, snapshot, size=768)[2]
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 8.0), dpi=100, facecolor="black")
    axes[0].imshow(image)
    clean_axis(axes[0])
    axes[1].imshow(target[2], cmap="gray", vmin=0, vmax=1, origin="lower")
    clean_axis(axes[1])
    axes[0].set_title(
        "Opaque and translucent surfaces", fontsize=16, color="#f0f3f7", pad=16
    )
    axes[1].set_title(
        "Light reaching the receiving surface", fontsize=16, color="#f0f3f7", pad=16
    )
    ax = axes[2]
    ax.set_facecolor("black")
    ax.tick_params(colors="#a7b1bd")
    ax.yaxis.label.set_color("#a7b1bd")
    ax.spines[["left", "bottom"]].set_color("#56606d")
    heights = [1, 0, 0.2, 0.7, 0.14]
    ax.bar(
        range(5),
        heights,
        color=["#c3cbd6", "#1f7abb", "#f26b26", "#39c47a", "#8c8252"],
        width=0.65,
    )
    ax.set_xticks(
        range(5), ["Clear", "Blue", "Orange", "Green", "Orange\n+ green"], fontsize=12
    )
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Transmittance", fontsize=12, color="#a7b1bd")
    ax.spines[["top", "right"]].set_visible(False)
    for i, h in enumerate(heights):
        ax.text(
            i, h + 0.045, f"{100 * h:g}%", ha="center", fontsize=13, color="#f0f3f7"
        )
    ax.set_title(
        "Each layer attenuates the light", fontsize=16, color="#f0f3f7", pad=16
    )
    fig.text(
        0.5,
        0.15,
        r"$T(q)=\prod_{f\prec q}(1-\alpha_f)$",
        ha="center",
        fontsize=21,
        color="#f0f3f7",
    )
    fig.text(
        0.5,
        0.03,
        "Opacity: blue 1.0, orange 0.8, green 0.3. "
        "Orange + green transmit 0.2 × 0.7 = 0.14; blue blocks the light.",
        ha="center",
        fontsize=13,
        color="#a7b1bd",
    )
    fig.subplots_adjust(left=0.025, right=0.98, bottom=0.29, top=0.89, wspace=0.18)
    fig.savefig(output / "source_features_sampling.png")
    plt.close(fig)


def make_figure(histories, snapshots, images, target):
    fig = plt.figure(figsize=(19.2, 16.0), dpi=100, facecolor="black")
    grid = fig.add_gridspec(
        3,
        3,
        left=0.035,
        right=0.975,
        top=0.98,
        bottom=0.06,
        height_ratios=[2.7, 1.15, 0.85],
        hspace=0.22,
        wspace=0.14,
    )
    renders = []
    fields = []
    for light in range(3):
        ax = fig.add_subplot(grid[0, light])
        renders.append(ax.imshow(images[0, light]))
        clean_axis(ax)
        ax.set_title("", fontsize=20, color="#f0f3f7", weight="bold", pad=12)
        sub = grid[1, light].subgridspec(1, 2, wspace=0.08)
        for col, data in enumerate((target[light], snapshots[0]["field"][light])):
            ax = fig.add_subplot(sub[0, col])
            h = ax.imshow(data, vmin=0, vmax=1, cmap="gray", origin="lower")
            clean_axis(ax)
            ax.set_title(
                "Target · receiver UV" if col == 0 else "Fitted · receiver UV",
                fontsize=13,
                color="#a7b1bd",
                pad=10,
            )
            if col == 1:
                fields.append(h)
    status = fig.add_subplot(grid[2, 0])
    status.axis("off")
    step_text = status.text(0.02, 0.88, "", fontsize=23, weight="bold", color="#f0f3f7")
    alpha_text = status.text(
        0.02, 0.31, "", fontsize=15, color="#f0f3f7", linespacing=1.5
    )
    status.text(0.02, 0.03, "Supervision: shadows only", fontsize=14, color="#a7b1bd")
    chart = fig.add_subplot(grid[2, 1:])
    lines = []
    for color, label in [
        ("#32c4e6", "With boundary gradients"),
        ("#f3a05b", "Smooth gradients only"),
    ]:
        lines.append(chart.plot([], [], color=color, label=label, lw=2.4)[0])
    all_losses = [r["loss"] for rows in histories.values() for r in rows]
    chart.set(
        yscale="log",
        xlim=(0, histories["boundary"][-1]["step"]),
        ylim=(max(min(all_losses) * 0.5, 1e-17), max(all_losses) * 2),
        xlabel="Optimization step",
        ylabel="UV transmittance MSE",
    )
    chart.set_facecolor("black")
    chart.tick_params(colors="#a7b1bd")
    chart.xaxis.label.set_color("#a7b1bd")
    chart.yaxis.label.set_color("#a7b1bd")
    chart.spines[["left", "bottom"]].set_color("#56606d")
    chart.spines[["top", "right"]].set_visible(False)
    chart.grid(alpha=0.2)
    chart.legend(
        frameon=False,
        fontsize=12,
        loc="lower right",
        bbox_to_anchor=(1, 1.02),
        ncol=2,
        labelcolor="#e2e6ec",
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
        a, b, c = row["opacity"]
        alpha_text.set_text(
            f"Blue opacity: {a:.3f}  (target 1.000)\n"
            f"Orange opacity: {b:.3f}  (target 0.800)\n"
            f"Green opacity: {c:.3f}  (target 0.300)"
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
            "1920x1600",
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
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=512)
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
    np.savez_compressed(
        args.output / "source_features_fit.npz",
        target=target,
        **{key: np.asarray([s[key] for s in snapshots]) for key in snapshots[0]},
    )
    (args.output / "source_features_demo.json").write_text(
        json.dumps({"checks": checks, "histories": histories}, indent=2)
    )
    print("Rendering scene views and optimization video...", flush=True)
    images = np.asarray([render_scene(scene, snapshot) for snapshot in snapshots])
    draw_explanation(args.output, scene, target)
    write_video(args.output, histories, snapshots, images, target, args.fps)


if __name__ == "__main__":
    main()

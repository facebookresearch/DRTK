# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import time

import cv2
import numpy as np
import torch as th
from drtk.renderlayer import RenderLayer

###
# This sanity check script should produce a sequence of images of a 2-colored
# square rendered on a random background whose vertices are optimized via
# differentiable rendering to match a target image.
#
# The renders should contain 3 rendered views of the square on the left, and
# corresponding difference images on the right. The difference images should
# approach 0 over the course of optimization. We print out the iteration, image
# loss, and geometry loss to the console. The geometry loss is not used during
# optimization, only image loss. The geometry loss should approach 0 using only
# image loss if the RL is working properly.
###


def gaussian_kernel(ksize, std=None):
    assert ksize % 2 == 1
    radius = ksize // 2
    if std is None:
        std = np.sqrt(-(radius**2) / (2 * np.log(0.05)))

    x, y = np.meshgrid(
        np.linspace(-radius, radius, ksize), np.linspace(-radius, radius, ksize)
    )
    xy = np.stack([x, y], axis=2)
    gk = np.exp(-(xy**2).sum(-1) / (2 * std**2))
    gk /= gk.sum()
    return gk


if __name__ == "__main__":
    if not os.path.exists("sanity_imgs"):
        os.mkdir("sanity_imgs")
    ksize = 3

    b = 3
    h = 1024
    w = 1024
    camrot = th.cat(
        [
            th.eye(3)[None].cuda(),
            th.from_numpy(cv2.Rodrigues(np.array([0, np.pi / 6, 0]))[0])
            .float()
            .cuda()[None],
            th.from_numpy(cv2.Rodrigues(np.array([0, -np.pi / 6, 0]))[0])
            .float()
            .cuda()[None],
        ],
        dim=0,
    )
    campos = th.cat(
        [
            th.FloatTensor([0, 0, -5])[None].cuda(),
            th.FloatTensor([0, 0, -5])[None].cuda(),
            th.FloatTensor([0, 0, -5])[None].cuda(),
        ],
        dim=0,
    )
    focal = th.FloatTensor([[w / 2, 0], [0, h / 2]])[None].cuda().expand(b, -1, -1)
    princpt = th.FloatTensor([w / 2, h / 2])[None].cuda().expand(b, -1)

    th.manual_seed(0)

    # Produce a fixed texture and an initial mesh to be optimized.
    tex = 255 * th.ones(1, 3, 512, 512).cuda()
    tex[:, :, :256] = th.FloatTensor([255, 128, 32]).cuda()[:, None, None]

    v = (
        th.FloatTensor(
            [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]]
        )[None].cuda()
        * 2
    )

    vt = th.FloatTensor(
        [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]
    )
    vi = th.IntTensor([[0, 1, 2], [3, 4, 5]])
    vti = th.IntTensor([[0, 1, 2], [3, 4, 5]])

    rl = RenderLayer(h, w, vt, vi, vti).cuda()

    # Verify that the RenderLayer is picklable.
    th.save(rl, "_tmp.pt")
    rl = th.load("_tmp.pt")
    os.unlink("_tmp.pt")

    # Generate the ground truth / target image by shifting the vertices in the
    # XY plane.
    shift = th.FloatTensor([0.25, 0.25, 0])[None, None].cuda()
    v_gt = v.clone() + shift
    v[:, :3] -= th.FloatTensor([0.3, 0.3, 0])[None].cuda()
    out = rl(
        v_gt.expand(b, -1, -1),
        tex.expand(b, -1, -1, -1),
        campos,
        camrot,
        focal,
        princpt,
        output_filters=["render", "mask"],
    )
    render = out["render"]
    mask = out["mask"]
    tgt = render * mask[:, None].float()
    bg_gt = 255 * th.rand(b, 3, 8, 8).cuda()
    _bg_gt = th.nn.functional.interpolate(
        bg_gt, scale_factor=128, mode="bilinear", align_corners=False
    )
    tgt[~mask[:, None].expand(-1, 3, -1, -1)] = _bg_gt[
        ~mask[:, None].expand(-1, 3, -1, -1)
    ]

    bg = bg_gt
    v = v[:, :, :2].clone().requires_grad_(True)
    lr = 5e-3

    pgrps = [{"params": [v], "lr": lr}]
    optim = th.optim.Adam(pgrps, lr=lr, betas=(0.9, 0.999))

    stack = th.cat([tgt[0], tgt[1], tgt[2]], dim=1)
    cv2.imwrite("sanity_imgs/target.png", stack.data.cpu().numpy().transpose(1, 2, 0))

    # Optimize the vertices using image loss.
    start = time.time()
    for it in range(1000):
        _bg = th.nn.functional.interpolate(
            bg, scale_factor=128, mode="bilinear", align_corners=False
        )
        _v = th.nn.functional.pad(v, (0, 1))
        out = rl(
            _v.expand(b, -1, -1),
            tex.expand(b, -1, -1, -1),
            campos,
            camrot,
            focal,
            princpt,
            background=_bg,
            ksize=ksize,
        )
        render = out["render"]

        diff = tgt - render
        loss = (diff**2).mean()

        if it == 0:
            stack = th.cat([render, (diff + 128) / 2], dim=3)
            stack = th.cat([stack[0], stack[1], stack[2]], dim=1)
            cv2.imwrite(
                f"sanity_imgs/{it:06d}.png", stack.data.cpu().numpy().transpose(1, 2, 0)
            )
            print(it, f"{loss:0.3e}")

        optim.zero_grad()
        loss.backward()
        optim.step()

        if it % 10 == 0 and it > 0:
            stack = th.cat([render, (diff + 128) / 2], dim=3)
            stack = th.cat([stack[0], stack[1], stack[2]], dim=1)
            cv2.imwrite(
                f"sanity_imgs/{it:06d}.png", stack.data.cpu().numpy().transpose(1, 2, 0)
            )
            gloss = ((_v - v_gt) ** 2).mean().item()

            spi = (time.time() - start) / it
            print(
                it,
                f"image loss (optimized): {loss:0.3e} geometry loss (not optimized): {gloss:0.3e}, time per iter(s): {spi:0.4f}",
            )

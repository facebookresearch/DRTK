# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import torch as th
import torch.nn.functional as thf
from drtk import edge_grad_estimator, interpolate, rasterize, render


def main(write_images, xy_only=False, z_only=False):
    assert not (xy_only and z_only), "You need to optimize at least some axes."

    v = th.tensor(
        [
            [10, 200, 100],
            [300, 50, 100],
            [400, 500, 100],
            [50, 400, 200],
            [400, 50, 50],
            [300, 500, 200],
        ],
        dtype=th.float32,
        device="cuda",
    )

    vt = th.zeros(1, 6, 2, device="cuda")
    vt[:, 3:6, 0] = 1

    vi = th.arange(6, device="cuda").int().view(2, 3)

    w = 512
    h = 512

    tex = th.ones(1, 3, 16, 16, device="cuda")
    tex[:, :, :, 8:] = 0.5
    v = th.nn.Parameter(v[None, ...])

    # Render the GT (target) image.
    with th.no_grad():
        v_gt = v.clone()
        th.cuda.manual_seed(10)
        v += th.randn_like(v) * 20.0

        index_img = rasterize(v_gt, vi, h, w)
        _, bary_img = render(v_gt, vi, index_img)
        vt_img = interpolate(vt, vi, index_img, bary_img).permute(0, 2, 3, 1)
        img_gt = (
            thf.grid_sample(tex, vt_img, padding_mode="border", align_corners=False)
            * (index_img != -1)[:, None]
        )

        img = (255 * img_gt[0]).clamp(0, 255).byte().data.cpu().numpy()
        cv2.imwrite("two_triangle_imgs/target.png", img.transpose(1, 2, 0)[..., ::-1])

    optim = th.optim.Adam([v], lr=1e-1, betas=(0.9, 0.999))

    # Optimize geometry to match target.
    for it in range(2000):
        index_img = rasterize(v, vi, h, w)
        _, bary_img = render(v, vi, index_img)
        vt_img = interpolate(vt, vi, index_img, bary_img).permute(0, 2, 3, 1)
        img = (
            thf.grid_sample(tex, vt_img, padding_mode="border", align_corners=False)
            * (index_img != -1)[:, None]
        )

        img = edge_grad_estimator(
            v_pix=v,
            vi=vi,
            bary_img=bary_img,
            img=img,
            index_img=index_img,
        )
        loss = ((img - img_gt) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        if xy_only:
            v.grad[..., 2] = 0
        if z_only:
            v.grad[..., :2] = 0
        optim.step()

        if it % 20 == 0:
            print(it, f"{loss.item():0.3e}")

            if write_images:
                img = (255 * img[0]).clamp(0, 255).byte().data.cpu().numpy()
                cv2.imwrite(
                    f"two_triangle_imgs/{it:06d}.png", img.transpose(1, 2, 0)[..., ::-1]
                )


if __name__ == "__main__":
    write_images = True
    if write_images and not os.path.exists("two_triangle_imgs"):
        os.mkdir("two_triangle_imgs")

    main(write_images)

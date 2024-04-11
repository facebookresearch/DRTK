# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

import cv2
import torch as th
from drtk.renderlayer import edge_grad_estimator, RenderLayer


def main(write_images, xy_only=False, z_only=False):
    assert not (xy_only and z_only), "You need to optimize at least some axes."

    v = th.tensor(
        [
            [0, 400, 0],
            [200, 100, 0],
            [400, 500, 0],
            [50, 300, -10],
            [400, 50, 20],
            [300, 500, 10],
        ],
        dtype=th.float32,
        device="cuda",
    )

    vt = th.zeros(6, 2, device="cuda")
    vt[3:6, 0] = 1

    vi = th.arange(6, device="cuda").int().view(2, 3)
    vti = vi.clone()

    w = 512
    h = 512
    camrot = th.eye(3)[None].cuda()
    camrot[0, 1, 1] = -1
    camrot[0, 2, 2] = -1
    campos = th.FloatTensor([256, 256, 700])[None].cuda()
    focal = th.FloatTensor([[w, 0], [0, h]])[None].cuda()
    princpt = th.FloatTensor([w / 2, h / 2])[None].cuda()

    tex = th.ones(1, 3, 16, 16, device="cuda")
    v = th.nn.Parameter(v[None, ...])

    rl = RenderLayer(h, w, vt, vi, vti).cuda()

    # Render the GT (target) image.
    with th.no_grad():
        tex[:, :, :, 8:] = 0
        v_gt = v.clone()
        v_gt[:, 3:] = -10000
        out_gt = rl(
            v_gt, tex, campos, camrot, focal, princpt, output_filters=["render"]
        )
        img_gt = out_gt["render"]
        tex[:, :, :, 8:] = 0.5

        img = (255 * img_gt[0]).clamp(0, 255).byte().data.cpu().numpy()
        cv2.imwrite("two_triangle_imgs/target.png", img.transpose(1, 2, 0)[..., ::-1])

    optim = th.optim.Adam([v], lr=1e-1, betas=(0.9, 0.999))
    output_filters = [
        "vt_img",
        "index_img",
        "render",
        "mask",
        "depth_img",
        "bary_img",
        "v_pix",
    ]

    # Optimize geometry to match target.
    for it in range(2000):
        out = rl(
            v,
            tex,
            campos,
            camrot,
            focal,
            princpt,
            output_filters=output_filters,
        )

        img = edge_grad_estimator(
            v_pix=out["v_pix"],
            vi=rl.vi,
            bary_img=out["bary_img"],
            img=out["render"],
            index_img=out["index_img"],
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

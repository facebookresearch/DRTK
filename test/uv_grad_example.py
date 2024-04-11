# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

import cv2
import drtk
import numpy as np
import torch as th
import torch.nn.functional as thf
from drtk.renderlayer import RenderLayer
from PIL import Image, ImageDraw


def main():
    """
    Renders a plane and visualizes uv grads -  du/dx, dv/dx, du/dy, dv/dy
    Displays comparison of analytically computed uv grads and numerically by taking finite differences
    Saves two variants: precise and approximate
    Precise: uv grads are computed per-pixel
    Approximate: uv grads are computed per-vertex and then interpolated linearly
    """

    s = 2000
    v = th.tensor(
        [
            [-s / 2, 0, -s],
            [s / 2, 0, -s],
            [s / 2, 0, 0],
            [-s / 2, 0, 0],
        ],
        dtype=th.float32,
        device="cuda",
    )
    vt = th.tensor(
        [
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0],
        ],
        dtype=th.float32,
        device="cuda",
    )
    vi = th.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=th.int32,
        device="cuda",
    )
    vti = vi.clone()

    checker = th.zeros(1, 3, 32, 32, dtype=th.float32, device="cuda")
    checker[:, :, 0::2, :] = 0.0
    checker[:, :, 1::2, :] = 1.0
    checker[:, :, :, 0::2] = 1.0 - checker[:, :, :, 1::2]
    checker = thf.interpolate(checker, (1024, 1024))

    w = 512
    h = 512
    camrot = th.eye(3)[None].cuda()
    a = 0.7
    camrot[0, 1, 1] = -np.cos(a)
    camrot[0, 1, 2] = np.sin(a)
    camrot[0, 2, 1] = -np.sin(a)
    camrot[0, 2, 2] = -np.cos(a)
    campos = th.FloatTensor([0, 1000, 500])[None].cuda()
    focal = th.FloatTensor([[w, 0], [0, h]])[None].cuda()
    princpt = th.FloatTensor([w / 2, h / 2])[None].cuda()

    rl = RenderLayer(h, w, vt, vi, vti).cuda()

    def render():
        out = rl(
            v[None, ...],
            checker,
            campos,
            camrot,
            focal,
            princpt,
            output_filters=[
                "render",
                "vt_dxdy_img",
                "vt_img",
            ],
        )
        img = out["render"]
        vt_dxdy_img = out["vt_dxdy_img"]
        vt_img = out["vt_img"]

        def diff(x, dim):
            s = list(x.shape)
            s[dim] = 1
            _x = th.cat([x, th.zeros(s, device=x.device, dtype=x.dtype)], dim=dim)
            _x = _x[(slice(None),) * dim + (slice(1, None),)]
            return _x - x

        du_dx = vt_dxdy_img[..., 0, 0][:, None].expand(-1, 3, -1, -1)
        dv_dx = vt_dxdy_img[..., 0, 1][:, None].expand(-1, 3, -1, -1)
        du_dy = vt_dxdy_img[..., 1, 0][:, None].expand(-1, 3, -1, -1)
        dv_dy = vt_dxdy_img[..., 1, 1][:, None].expand(-1, 3, -1, -1)

        img_row = np.concatenate(
            [
                add_text(convert_image(im), 10, 512 - 20, text)
                for im, text in zip(
                    [img, du_dx * 200, dv_dx * 200, du_dy * 200 + 0.5, dv_dy * 50],
                    ["render", "du/dx", "dv/dx", "du/dy", "dv/dy"],
                )
            ],
            axis=1,
        )

        vt_img = vt_img * 0.5 + 0.5
        du_dx_num = diff(vt_img[..., 0], 2)[:, None].expand(-1, 3, -1, -1)
        dv_dx_num = diff(vt_img[..., 1], 2)[:, None].expand(-1, 3, -1, -1)
        du_dy_num = diff(vt_img[..., 0], 1)[:, None].expand(-1, 3, -1, -1)
        dv_dy_num = diff(vt_img[..., 1], 1)[:, None].expand(-1, 3, -1, -1)

        img_row_num = np.concatenate(
            [
                add_text(convert_image(im), 10, 512 - 20, text)
                for im, text in zip(
                    [
                        img,
                        du_dx_num * 200,
                        dv_dx_num * 200,
                        du_dy_num * 200 + 0.5,
                        dv_dy_num * 50,
                    ],
                    [
                        "render",
                        "du/dx finite diff",
                        "dv/dx finite diff",
                        "du/dy finite diff",
                        "dv/dy finite diff",
                    ],
                )
            ],
            axis=1,
        )

        img = np.concatenate([img_row, img_row_num], axis=0)
        return img

    img_approximate = render()
    drtk.renderlayer.settings.use_precise_uv_grads = True
    img_precise = render()

    cv2.imwrite("uv_grad_test/uv_grad_approximate.png", img_approximate[..., ::-1])
    cv2.imwrite("uv_grad_test/uv_grad_precise.png", img_precise[..., ::-1])


def convert_image(img):
    return (255 * img[0]).clamp(0, 255).byte().data.cpu().numpy().transpose(1, 2, 0)


def add_text(im, x, y, text):
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    draw.text((x, y), text, (255, 255, 255))
    return np.asarray(im)


if __name__ == "__main__":
    os.makedirs("uv_grad_test", exist_ok=True)
    main()

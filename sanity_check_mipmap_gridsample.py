import os

from typing import List

import cv2
import drtk
import numpy as np

import torch as th
import torch.nn.functional as thf
from drtk.renderlayer import mipmap_grid_sample, RenderLayer
from drtk.renderlayer.mipmap_grid_sampler_ref_impl import (
    mipmap_grid_sample as mipmap_grid_sample_ref,
)
from PIL import Image, ImageDraw

max_mipmap_levels = 4
inter_mode = "bilinear"
padding_mode = "border"

aniso_levels = [1, 2, 4, 8, 16, 32]

# ref implementation does not support grad clipping, so
# for comparison with ref implementation we disable gradient clipping
# The gradient clipping only effect the case of anisotropic filtering when not full mipmap pyramid is
# available.
# This gradient is uv gradient used for determining mipmap level and anisotropicity (it is not the gradient as
# in backward pass)
# Please note that hardware typically requires the full pyramid to be present.
clip_grad = False

# ref implementation always forces maximum anisotropic filtration, but that is due to the
# limitations of the implementation.
# Forcing maximum anisotropic filtration significantly degrades speed of the CUDA kernel
# Actual hardware operates similar to when `force_max_aniso = False`
# `mipmap_grid_sample` is not supposed to be used with force_max_aniso=True, that flag is only for
# comparing it with the ref implementation
# When `force_max_aniso = False`, there will be slight divergence between CUDA and ref implementation.
force_max_aniso = False


def make_pyramid_average_2x2(
    texture: th.Tensor, max_mipmap_levels: int
) -> List[th.Tensor]:
    """
    See p. 266: 8.14.4    Manual Mipmap Generation
    No particular filter algorithm is required, though a box filter is recommended as the default filter
    """

    def get_level_number_for_texture(texture: th.Tensor) -> int:
        return min(max_mipmap_levels - 1, int(np.log2(min(*texture.shape[-2:]))))

    mipmap_levels = get_level_number_for_texture(texture)
    result = [texture]
    for _ in range(mipmap_levels):
        texture = thf.avg_pool2d(texture, kernel_size=2, stride=2)
        result.append(texture)
    return result


def main():
    s = 6100
    v = th.tensor(
        [
            [-s / 2 * 0.707, s / 2 * 0.707, -s],
            [0, 0, -s],
            [0, 0, 0],
            [-s / 2 * 0.707, s / 2 * 0.707, 0],
            [s / 2, 0, -s],
            [s / 2, 0, 0],
        ],
        dtype=th.float32,
        device="cuda",
    )
    vt = th.tensor(
        [
            [0, 1],
            [0.5, 1],
            [0.5, 0],
            [0, 0],
            [1, 1],
            [1, 0],
        ],
        dtype=th.float32,
        device="cuda",
    )
    vi = th.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [1, 2, 5],
            [1, 5, 4],
        ],
        dtype=th.int32,
        device="cuda",
    )
    vti = vi.clone()

    def make_checker() -> th.Tensor:
        def make_grid(n: int, size: int) -> th.Tensor:
            grid = th.zeros(1, 3, n, n, dtype=th.float32, device="cuda")
            grid[:, :, 1::2, :] = 1.0
            grid[:, :, :, 0::2] = 1.0 - grid[:, :, :, 1::2]
            return thf.interpolate(grid, (size, size))

        g0 = make_grid(1024, 1024)
        g1 = make_grid(256, 1024)
        g2 = make_grid(32, 1024)
        g3 = make_grid(8, 1024)
        checker = 0.5 - 0.02 * g0 - 0.03 * g1 - 0.04 * g2 - 0.05 * g3
        checker[:, :, ::32, :] += 0.3
        checker[:, :, :, ::32] += 0.3
        checker[:, :, ::32, ::32] -= 0.3
        return checker

    checker = make_checker()

    batch = 2

    w = 1024
    h = 1024
    camrot = th.eye(3)[None].cuda()
    a = 0.4
    camrot[0, 1, 1] = -np.cos(a)
    camrot[0, 1, 2] = np.sin(a)
    camrot[0, 2, 1] = -np.sin(a)
    camrot[0, 2, 2] = -np.cos(a)
    campos = th.FloatTensor([0, 1000, 500])[None].cuda()
    focal = th.FloatTensor([[w / 2, 0], [0, h / 2]])[None].cuda()
    princpt = th.FloatTensor([w / 2, h / 2])[None].cuda()

    rl = RenderLayer(h, w, vt, vi, vti).cuda()

    drtk.renderlayer.settings.use_precise_uv_grads = True

    out = rl(
        v[None, ...].expand(batch, -1, -1),
        checker.expand(batch, -1, -1, -1),
        campos.expand(batch, -1),
        camrot.expand(batch, -1, -1),
        focal.expand(batch, -1, -1),
        princpt.expand(batch, -1),
        output_filters=[
            "render",
            "vt_dxdy_img",
            "vt_img",
            "mask",
        ],
    )
    vt_dxdy_img = out["vt_dxdy_img"]
    vt_img = out["vt_img"]
    img_raw = out["render"]
    mask = out["mask"]

    output_img_cuda = []
    output_img_pytorch = []
    output_img_pytorch_hq = []
    output_grid_grad_pytorch = []
    output_grid_grad_cuda = []
    output_input_grad_pytorch = []
    output_input_grad_cuda = []
    output_img_diff = []
    output_grid_grad_diff = []
    output_input_grad_diff = []

    checker_pyramid = make_pyramid_average_2x2(
        checker.expand(batch, -1, -1, -1), max_mipmap_levels=max_mipmap_levels
    )
    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)

    checker_pyramid = [x.requires_grad_(True) for x in checker_pyramid]
    vt_img_g = vt_img.requires_grad_(True)
    vt_dxdy_img = vt_dxdy_img.requires_grad_(False)

    for x in aniso_levels:
        start.record()
        img_pytorch = mipmap_grid_sample_ref(
            checker_pyramid,
            vt_img_g,
            vt_dxdy_img,
            max_aniso=x,
            mode=inter_mode,
            padding_mode=padding_mode,
        )
        end.record()
        th.cuda.synchronize()
        print(f"pytorch {start.elapsed_time(end)}")

        grad = img_pytorch * 0.0005 * mask[:, None]
        start.record()
        img_pytorch.backward(grad)
        end.record()
        th.cuda.synchronize()
        print(f"backward: pytorch {start.elapsed_time(end)}")

        grid_grad_pytorch = vt_img_g.grad
        input_grad_pytorch = merge_pyramid(
            [x.grad * 100.0 + 0.5 for x in checker_pyramid]
        )

        output_img_pytorch.append(img_pytorch * mask[:, None])
        output_grid_grad_pytorch.append(grid_grad_pytorch + 0.5)
        output_input_grad_pytorch.append(input_grad_pytorch)

        vt_img_g.grad = None
        for m in checker_pyramid:
            m.grad = None

        # High quality mipmap filtering, for reference
        with th.no_grad():
            img_pytorch_hq = mipmap_grid_sample_ref(
                checker_pyramid,
                vt_img_g,
                vt_dxdy_img,
                max_aniso=x,
                mode=inter_mode,
                padding_mode=padding_mode,
                high_quality=True,
            )
            output_img_pytorch_hq.append(img_pytorch_hq * mask[:, None])

        start.record()
        img_cuda = mipmap_grid_sample(
            checker_pyramid,
            vt_img_g,
            vt_dxdy_img,
            max_aniso=x,
            mode=inter_mode,
            force_max_aniso=force_max_aniso,
            padding_mode=padding_mode,
            clip_grad=clip_grad,
        )
        end.record()
        th.cuda.synchronize()
        print(f"cuda {start.elapsed_time(end)}")

        grad = img_cuda * 0.0005 * mask[:, None]
        start.record()
        img_cuda.backward(grad)
        end.record()
        th.cuda.synchronize()
        print(f"backward: cuda {start.elapsed_time(end)}")

        grid_grad_cuda = vt_img_g.grad
        input_grad_cuda = merge_pyramid([x.grad * 100.0 + 0.5 for x in checker_pyramid])

        output_img_cuda.append(img_cuda * mask[:, None])
        output_grid_grad_cuda.append(grid_grad_cuda + 0.5)
        output_input_grad_cuda.append(input_grad_cuda)

        vt_img_g.grad = None
        for m in checker_pyramid:
            m.grad = None

        output_img_diff.append(th.abs(img_pytorch - img_cuda) + 0.5)
        output_grid_grad_diff.append(th.abs(grid_grad_pytorch - grid_grad_cuda) + 0.5)
        output_input_grad_diff.append(
            th.abs(input_grad_pytorch - input_grad_cuda) + 0.5
        )

    output_img_cuda = compose_horizontal(
        [img_raw] + output_img_cuda,
        ["img_raw"] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_img_pytorch = compose_horizontal(
        [img_raw] + output_img_pytorch,
        ["img_raw"] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_img_pytorch_hq = compose_horizontal(
        [img_raw] + output_img_pytorch_hq,
        ["img_raw"] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_img_diff = compose_horizontal(
        [th.zeros_like(img_raw)] + output_img_diff,
        [""] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_grid_grad_pytorch = compose_horizontal(
        [th.zeros_like(img_raw)] + output_grid_grad_pytorch,
        [""] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_grid_grad_cuda = compose_horizontal(
        [th.zeros_like(img_raw)] + output_grid_grad_cuda,
        [""] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_grid_grad_diff = compose_horizontal(
        [th.zeros_like(img_raw)] + output_grid_grad_diff,
        [""] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_input_grad_pytorch = compose_horizontal(
        [th.zeros_like(img_raw)] + output_input_grad_pytorch,
        [""] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_input_grad_cuda = compose_horizontal(
        [th.zeros_like(img_raw)] + output_input_grad_cuda,
        [""] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    output_input_grad_diff = compose_horizontal(
        [th.zeros_like(img_raw)] + output_input_grad_diff,
        [""] + [f"max_aniso:{x}" for x in aniso_levels],
    )

    cv2.imwrite("mipmap_test/output_cuda.png", output_img_cuda[..., ::-1])
    cv2.imwrite("mipmap_test/output_pytorch.png", output_img_pytorch[..., ::-1])
    cv2.imwrite("mipmap_test/output_pytorch_hq.png", output_img_pytorch_hq[..., ::-1])

    cv2.imwrite("mipmap_test/output_diff.png", output_img_diff[..., ::-1])
    cv2.imwrite(
        "mipmap_test/grid_grad_pytorch.png", output_grid_grad_pytorch[..., ::-1]
    )
    cv2.imwrite("mipmap_test/grid_grad_cuda.png", output_grid_grad_cuda[..., ::-1])
    cv2.imwrite("mipmap_test/grid_grad_diff.png", output_grid_grad_diff[..., ::-1])
    cv2.imwrite(
        "mipmap_test/input_grad_pytorch.png", output_input_grad_pytorch[..., ::-1]
    )
    cv2.imwrite("mipmap_test/input_grad_cuda.png", output_input_grad_cuda[..., ::-1])
    cv2.imwrite("mipmap_test/input_grad_diff.png", output_input_grad_diff[..., ::-1])


def convert_image(img):
    if img.shape[3] < 3:
        img = img.permute(0, 3, 1, 2)
    if img.shape[1] == 2:
        img = th.cat([img, th.zeros_like(img[:, :1])], dim=1)
    return (255 * img[0]).clamp(0, 255).byte().data.cpu().numpy().transpose(1, 2, 0)


def add_text(im, x, y, text):
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    draw.text((x, y), text, (255, 255, 255))
    return np.asarray(im)


def compose_horizontal(img_list, text_list):
    return np.concatenate(
        [
            add_text(convert_image(im), 10, im.shape[-1] - 20, text)
            for im, text in zip(img_list, text_list)
        ],
        axis=1,
    )


def merge_pyramid(x):
    def _merge(head, tail):
        head_2, tail = tail[0], tail[1:]
        pad = th.zeros(
            *head.shape[:2],
            head.shape[2] - head_2.shape[2],
            head_2.shape[3],
            device=head.device,
        )
        head = th.cat([head, th.cat([head_2, pad], dim=2)], dim=3)
        if len(tail) == 0:
            return head
        else:
            return _merge(head, tail)

    head, tail = x[0], x[1:]
    if len(tail) == 0:
        return head
    return _merge(head, tail)


if __name__ == "__main__":
    os.makedirs("mipmap_test", exist_ok=True)
    main()

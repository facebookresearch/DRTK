# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch as th
import torch.nn as nn
from care.strict.utils.torch import index
from torch import Tensor


class PythonRenderer(nn.Module):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, h, w) -> None:
        super(PythonRenderer, self).__init__()
        pixelcrdy, pixelcrdx = th.meshgrid(th.arange(h), th.arange(w))
        self.register_buffer("pixelcrdx", pixelcrdx, persistent=False)
        self.register_buffer("pixelcrdy", pixelcrdy, persistent=False)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, v2d, vt: Tensor, vi: Tensor, vti: Tensor, index_img, vn=None):
        b = v2d.shape[0]
        mask = th.ne(index_img, -1)
        float_mask = mask.float()[:, None]
        iimg_clamped = index_img.clamp(min=0).long()

        # compute barycentric coordinates
        vi_img = index(vi, iimg_clamped, 0).long()
        v2d_img0 = th.cat(
            [index(v2d[i], vi_img[i, ..., 0].data, 0)[None, ...] for i in range(b)],
            dim=0,
        )
        v2d_img1 = th.cat(
            [index(v2d[i], vi_img[i, ..., 1].data, 0)[None, ...] for i in range(b)],
            dim=0,
        )
        v2d_img2 = th.cat(
            [index(v2d[i], vi_img[i, ..., 2].data, 0)[None, ...] for i in range(b)],
            dim=0,
        )

        vec01 = v2d_img1 - v2d_img0
        vec02 = v2d_img2 - v2d_img0

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def epsclamp(x):
            return th.where(x < 0, x.clamp(max=-1e-8), x.clamp(min=1e-8))

        det = vec01[..., 0] * vec02[..., 1] - vec01[..., 1] * vec02[..., 0]
        denom = epsclamp(det)

        px = self.pixelcrdx[None, ...] - v2d_img0[..., 0]
        py = self.pixelcrdy[None, ...] - v2d_img0[..., 1]

        lambda_1 = (px * vec02[..., 1] - py * vec02[..., 0]) / denom
        lambda_2 = (py * vec01[..., 0] - px * vec01[..., 1]) / denom
        lambda_0 = 1 - lambda_1 - lambda_2

        w0 = 1.0 / epsclamp(v2d_img0[:, :, :, 2])
        w1 = 1.0 / epsclamp(v2d_img1[:, :, :, 2])
        w2 = 1.0 / epsclamp(v2d_img2[:, :, :, 2])
        zi = 1.0 / epsclamp(w0 * lambda_0 + w1 * lambda_1 + w2 * lambda_2)

        bary_0 = w0 * lambda_0 * zi
        bary_1 = w1 * lambda_1 * zi
        bary_2 = w2 * lambda_2 * zi

        bary_img = th.cat(
            # pyre-fixme[16]: `float` has no attribute `__getitem__`.
            (bary_0[:, :, :, None], bary_1[:, :, :, None], bary_2[:, :, :, None]),
            dim=3,
        ) * float_mask.permute(0, 2, 3, 1)

        # compute texture coordinates
        vti_img = index(vti, iimg_clamped, 0).long()
        if len(vt.shape) == 3:
            vt_img0 = th.stack([index(vt[i], vti_img[i, ..., 0], 0) for i in range(b)])
            vt_img1 = th.stack([index(vt[i], vti_img[i, ..., 1], 0) for i in range(b)])
            vt_img2 = th.stack([index(vt[i], vti_img[i, ..., 2], 0) for i in range(b)])
        else:
            vt_img0 = index(vt, vti_img[..., 0].data, 0)
            vt_img1 = index(vt, vti_img[..., 1].data, 0)
            vt_img2 = index(vt, vti_img[..., 2].data, 0)

        vt_img = (
            vt_img0 * bary_0[:, :, :, None]
            + vt_img1 * bary_1[:, :, :, None]
            + vt_img2 * bary_2[:, :, :, None]
        )

        depth_img = zi * float_mask[:, 0]
        vt_img = th.lerp(
            # Filling unused vt with continuous span of UV space instead of zeros improves backwards pass
            # performance for subsequent grid_sample operations that use vt_img.
            th.stack(
                [
                    (self.pixelcrdx[None, ...] * 2 + 1) / self.pixelcrdx.shape[-1],
                    (self.pixelcrdy[None, ...] * 2 + 1) / self.pixelcrdx.shape[-2],
                ],
                dim=-1,
            )
            - 1,
            vt_img * 2 - 1,
            float_mask.permute(0, 2, 3, 1),
        )

        vn_img = None
        if vn is not None:
            vi_img = vi[iimg_clamped, :].long()
            vn_img0 = th.stack([index(vn[i], vi_img[i, ..., 0], 0) for i in range(b)])
            vn_img1 = th.stack([index(vn[i], vi_img[i, ..., 1], 0) for i in range(b)])
            vn_img2 = th.stack([index(vn[i], vi_img[i, ..., 2], 0) for i in range(b)])
            vn_img = (
                th.stack([vn_img0, vn_img1, vn_img2], dim=4) * bary_img[:, :, :, None]
            ).sum(-1)
            vn_img = vn_img * float_mask.permute(0, 2, 3, 1)

        return depth_img, bary_img, vt_img, vn_img

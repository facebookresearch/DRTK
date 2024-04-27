# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import numpy as np
import torch as th
from drtk.renderlayer import RenderLayer
from numpy.testing import assert_allclose


def test_near_far(far, near, dist=5, h=10, w=10, b=1, eps=1e-5):
    camrot = th.eye(3, device="cuda")[None].expand(b, -1, -1)
    campos = th.tensor([0, 0, -dist], dtype=th.float32, device="cuda")[None].expand(
        b, -1
    )
    focal = th.tensor([[w / 2, 0], [0, h / 2]], dtype=th.float32, device="cuda")[
        None
    ].expand(b, -1, -1)
    princpt = th.tensor([w / 2, h / 2], dtype=th.float32, device="cuda")[None].expand(
        b, -1
    )
    th.manual_seed(0)

    tex = 255 * th.ones(1, 3, 1, 1, device="cuda")

    v = (
        th.tensor(
            [
                [-1 - eps, 1, 0],
                [1, 1, 0],
                [1, -1 - eps, 0],
                [1 + eps, -1, 0],
                [-1, -1, 0],
                [-1, 1 + eps, 0],
            ],
            dtype=th.float32,
            device="cuda",
        )[None]
        * 100.0
    )

    vt = th.zeros((6, 2))
    vi = th.tensor([[0, 1, 2], [3, 4, 5]], dtype=th.int64)
    vti = th.tensor([[0, 1, 2], [3, 4, 5]], dtype=th.int64)

    rl = RenderLayer(h, w, vt, vi, vti).cuda()

    out = rl(
        v.expand(b, -1, -1),
        tex.expand(b, -1, -1, -1),
        campos,
        camrot,
        focal,
        princpt,
        far=far,
        near=near,
        output_filters=["render"],
    )
    return out["render"].detach().cpu().numpy()


class DRTKNearFarTests(unittest.TestCase):
    def test_near_far(self):
        full = np.full((1, 3, 10, 10), fill_value=255.0)
        empty = np.zeros((1, 3, 10, 10))

        out = test_near_far(far=None, near=None)
        assert_allclose(out, full)

        out = test_near_far(far=6.0, near=5.5)
        assert_allclose(out, empty)

        out = test_near_far(far=4.5, near=4.0)
        assert_allclose(out, empty)


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch as th
from care.strict.utils.geom import vert_normals
from renderlayer import RenderLayer, settings


def test_cuda_vs_py(cons_args, inputs):
    inputs[0].requires_grad_(True)
    inputs[1].requires_grad_(True)

    filters = ["render", "mask", "vt_img", "depth_img", "v_pix", "vn_img"]

    rl = RenderLayer(*cons_args).cuda()
    vn = vert_normals(inputs[0], rl.vi[None].long())
    out = rl(*inputs, output_filters=filters, vn=vn)
    abs(out["render"]).sum().backward(retain_graph=True)
    abs(out["vn_img"]).sum().backward(retain_graph=True)
    abs(out["depth_img"]).sum().backward()

    gg = inputs[0].grad.clone()
    gt = inputs[1].grad.clone()
    inputs[0].grad = None
    inputs[1].grad = None

    settings.use_python_renderer = True
    pyrl = RenderLayer(*cons_args).cuda()
    vn = vert_normals(inputs[0], rl.vi[None].long())
    pyout = pyrl(*inputs, output_filters=filters, vn=vn)
    abs(pyout["render"]).sum().backward(retain_graph=True)
    abs(pyout["vn_img"]).sum().backward(retain_graph=True)
    abs(pyout["depth_img"]).sum().backward()

    pygg = inputs[0].grad.clone()
    pygt = inputs[1].grad.clone()
    inputs[0].grad = None
    inputs[1].grad = None

    print("=== [Output mean abs. differences (CUDA vs. Python)] ===")
    for name, out in out.items():
        if name in pyout:
            print(
                f"{name:<32s}:",
                abs(out.float() - pyout[name].float()).mean().item(),
            )

    print()
    print(f"{'Geom. grad diff:':<32s}", abs(gg - pygg).mean().item())
    print(f"{'Tex. grad diff:':<32s}", abs(gt - pygt).mean().item())
    print(
        f"{'# Geom. grad sign diffs:':<32s}",
        (th.sign(gg) != th.sign(pygg)).sum().item(),
    )
    print(
        f"{'# Tex. grad sign diffs:':<32s}", (th.sign(gt) != th.sign(pygt)).sum().item()
    )


if __name__ == "__main__":
    cons_args = list(th.load("/mnt/projects/drtk/test_data/rl_cons_args.pt"))
    inputs = th.load("/mnt/projects/drtk/test_data/rl_inputs.pt")
    test_cuda_vs_py(deepcopy(cons_args), deepcopy(inputs))

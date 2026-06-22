# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import torch as th
import torch.nn.functional as thf
from drtk.utils import load_torch_ops

load_torch_ops("drtk.grid_scatter_ext")


@th.compiler.disable
def grid_scatter(
    input: th.Tensor,
    grid: th.Tensor,
    output_height: int,
    output_width: int,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: Optional[bool] = None,
) -> th.Tensor:
    """Scatter an image through a normalized sampling grid.

    ``grid_scatter`` is the splatting counterpart of
    :func:`torch.nn.functional.grid_sample`. In ``grid_sample``, each output
    pixel reads from a source location in the input. In ``grid_scatter``, each
    input pixel writes its value to a destination location described by
    ``grid``. The operation is useful for tasks such as projecting camera-view
    values into a UV texture atlas or accumulating visibility weights in
    texture space.

    The forward pass of ``grid_scatter`` corresponds to the input-gradient
    accumulation performed by ``grid_sample``. The backward pass correspondingly
    samples gradients back from the output grid, while also producing gradients
    with respect to ``grid``.

    Args:
        input: Source values with shape ``(N, C, H, W)``.
        grid: Destination coordinates with shape ``(N, H, W, 2)``. Coordinates
            use the same normalized ``[-1, 1]`` convention, padding modes, and
            ``align_corners`` semantics as PyTorch ``grid_sample``.
        output_height: Height of the scattered output image.
        output_width: Width of the scattered output image.
        mode: Interpolation kernel. Supported values are ``"bilinear"`` and
            ``"bicubic"``.
        padding_mode: Handling for samples outside the output image. Supported
            values are ``"zeros"``, ``"border"``, and ``"reflection"``.
        align_corners: Same meaning as in ``grid_sample``. ``None`` defaults to
            ``False``.

    Returns:
        Tensor with shape ``(N, C, output_height, output_width)`` containing the
        accumulated scattered values.

    Note:
        Multiple input pixels can scatter to the same output pixel; their
        weighted contributions are accumulated. This operation currently
        requires the accelerator backend.
    """
    if mode != "bilinear" and mode != "bicubic":
        raise ValueError(
            "grid_scatter(): only 'bilinear' and 'bicubic' modes are supported "
            "but got: '{}'".format(mode)
        )
    if (
        padding_mode != "zeros"
        and padding_mode != "border"
        and padding_mode != "reflection"
    ):
        raise ValueError(
            "grid_scatter(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            "but got: '{}'".format(padding_mode)
        )
    if mode == "bilinear":
        mode_enum = 0
    elif mode == "nearest":  # not supported
        mode_enum = 1
    else:  # mode == 'bicubic'
        mode_enum = 2

    if padding_mode == "zeros":
        padding_mode_enum = 0
    elif padding_mode == "border":
        padding_mode_enum = 1
    else:  # padding_mode == 'reflection'
        padding_mode_enum = 2

    if align_corners is None:
        align_corners = False

    return th.ops.grid_scatter_ext.grid_scatter_2d(
        input,
        grid,
        output_height,
        output_width,
        padding_mode_enum,
        mode_enum,
        align_corners,
    )


class GridScatterRef(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: th.Tensor,
        grid: th.Tensor,
        output_height: int,
        output_width: int,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: Optional[bool] = None,
    ):
        with th.enable_grad():
            tex = th.ones(
                input.shape[0],
                input.shape[1],
                output_height,
                output_width,
                dtype=input.dtype,
                device=input.device,
            )
            tex.requires_grad_(True)

            out = thf.grid_sample(
                tex,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )

            out.backward(input)
            ctx.save_for_backward(input, grid, out)
            ctx.mode = mode
            ctx.padding_mode = padding_mode
            ctx.align_corners = align_corners
            return tex.grad

    @staticmethod
    def backward(ctx, grad_output: th.Tensor):
        input, grid, out = ctx.saved_tensors
        grid = grid.clone().detach()
        grid.requires_grad_(True)

        with th.enable_grad():
            input_grad = thf.grid_sample(
                grad_output,
                grid,
                mode=ctx.mode,
                padding_mode=ctx.padding_mode,
                align_corners=ctx.align_corners,
            )
            input_grad.backward(input)

        return input_grad, grid.grad, None, None, None, None, None


_grid_scatter_ref = GridScatterRef.apply


@th.compiler.disable
def grid_scatter_ref(
    input: th.Tensor,
    grid: th.Tensor,
    output_height: int,
    output_width: int,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: Optional[bool] = None,
) -> th.Tensor:
    """Pure PyTorch reference implementation used by tests.

    This helper is intentionally not part of the documented public API. See
    :func:`drtk.grid_scatter` for the supported implementation.
    """
    return _grid_scatter_ref(
        input,
        grid,
        output_height,
        output_width,
        mode,
        padding_mode,
        align_corners,
    )

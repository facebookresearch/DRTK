from typing import List, Optional

import torch as th
from drtk import mipmap_grid_sampler_ext

th.ops.load_library(mipmap_grid_sampler_ext.__file__)


def mipmap_grid_sample(
    input: List[th.Tensor],
    grid: th.Tensor,
    vt_dxdy_img: th.Tensor,
    max_aniso: int,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
    force_max_aniso: Optional[bool] = False,
    clip_grad: Optional[bool] = False,
) -> th.Tensor:
    """
    Similar to torch.nn.functional.grid_sample, but supports mipmapping and anisotropic filtering which mimics
    graphics hardware behaviour.
    Currently, only spatial (4-D) inputs are supported.
    No nearest filtering.

    Args:
        input (List[th.Tensor]): A list of tensors which represents a mipmap pyramid of a texture. List should
            contain highest resolution texture at index 0.
            All subsequent elements of the list should contain miplayers that are twice smaller, but is not a
            hard requirement. Also there is no hard requirement for all mip levels to be present.
            List of tensors of shape [N x C x H_in x W_in], [N x C x H_in / 2 x W_in / 2] ... [N x C x 1 x 1]

        grid (th.Tensor): uv coordinates field according to which the inputs are sampled.
            N x H_out x W_out x 2

        vt_dxdy_img (th.Tensor): Jacobian of uv coordinates field with respect to pixel position.
            N x H_out x W_out x 2 x 2

        max_aniso (int): Maximum number of samples for anisotropic filtering.

        mode (str): Interpolation mode to calculate output values. Same as grid_sample, but without 'nearest'.
            'bilinear' | 'bicubic'. Default: 'bilinear'

        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'

        align_corners (bool, optional): Same as in grid_sample. Default: False.

        force_max_aniso (bool, optional): Contols number of samples for anisotropic filtering.
            When it is False, the extension will work similarly to the graphics hardware implementation,
            e.i. it does only needed number of samples, which could be anything from 1 to max_aniso depending
            on the ratio of max and min uv gradient. When force_max_aniso is True, the extension always
            produces max_aniso number of samples. However this mode is only intended for
            debugging/tesing/comparing with reference implementation and it is not intended for the real
            usage. Default: False.

        clip_grad (bool, optional): Controls behaviour when mipmap layer is missing.
            This mipmap implementation allows using not full mipmap pyramid, e.g. you can have only 2, 3 or 4
            layers for a texture of size 1024 instead of all 10. Hardware require all of the layers of the
            pyramid to be present. Such relaxed requirement leads to ambiguity when it needs to sample from
            the missing layer. The flag clip_grad only impacts the cases when needed layers from the mipmap
            pyramid are missing. In this scenario:
                - when False: it will sample from the last available layer. This will lead to aliasing and
                  sparsely placed taps. The downside is that it may sample from arbitrary far regions of
                  the texture.
                - when True: it will sample from the last available layer, but it will adjust the step size
                  to match the sampling rate of the available layer. This will lead to aliasing but densely
                  placed taps. Which in turn forbids it to sample from arbitrary far regions of the texture.

    Returns:
        output (Tensor): Result of sampling from inputs given the grid.
            N x C x H_out x W_out
    """

    if mode != "bilinear" and mode != "bicubic":
        raise ValueError(
            "mipmap_grid_sample(): only 'bilinear' and 'bicubic' modes are supported "
            "but got: '{}'".format(mode)
        )
    if (
        padding_mode != "zeros"
        and padding_mode != "border"
        and padding_mode != "reflection"
    ):
        raise ValueError(
            "mipmap_grid_sample(): expected padding_mode "
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

    return th.ops.mipmap_grid_sampler_ext.mipmap_grid_sampler_2d(
        input,
        grid,
        vt_dxdy_img,
        max_aniso,
        padding_mode_enum,
        mode_enum,
        align_corners,
        force_max_aniso,
        clip_grad,
    )

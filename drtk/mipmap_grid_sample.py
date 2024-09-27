# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch as th
import torch.nn.functional as thf
from drtk.utils import load_torch_ops

load_torch_ops("drtk.mipmap_grid_sampler_ext")


@th.compiler.disable
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


def mipmap_grid_sample_ref(
    input: List[th.Tensor],
    grid: th.Tensor,
    vt_dxdy_img: th.Tensor,
    max_aniso: int,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: Optional[bool] = False,
    high_quality: bool = False,
) -> th.Tensor:
    """
    A reference implementation for `mipmap_grid_sample`. See the doc string from `mipmap_grid_sample`
    The CUDA version of `mipmap_grid_sample` should behave the same as this referense implementation when:
        - `force_max_aniso` argument of `mipmap_grid_sample` is set to True
        - `clip_grad` argument of `mipmap_grid_sample` is set to False
        - `high_quality` argument of `mipmap_grid_sample_ref` is set to False
    """

    q = len(input)
    base_level_size = list(input[0].shape[2:])

    with th.no_grad():
        # vt_dxdy_img has assumes uv in range 0..1.
        # For the comutations below we need to convert from normalized units to pixels
        size = th.as_tensor(
            [base_level_size[0], base_level_size[1]],
            dtype=th.float32,
            device=vt_dxdy_img.device,
        )
        vt_dxdy_img_pixel = vt_dxdy_img * size[None, None, None, :]

        # x and y gradients magnitudes. We then need to find direction of maximum gradient and
        # minimum gradients direction (principal axis)
        px, py = _compute_grad_magnitude(vt_dxdy_img_pixel)

        if not high_quality:
            # This is what hardware implements.
            # We assume that maximum and minimum direction is either x or y.
            # This assumption is a quite grude approximation
            p_max = th.max(px, py)
            p_min = th.min(px, py) if max_aniso != 1 else None
        else:
            # Instead, a more correct way would be to find principal axis using SVD
            # Note this is not practical as it is very slow
            u, s, v = th.linalg.svd(vt_dxdy_img_pixel)
            p_max = s[..., 0]
            p_min = s[..., 1]

        # Given the max and min gradients, select mipmap levels (assumes linear interpolation
        # between mipmaps)
        d1, a = _mipmap_selection(q, p_max, p_min, max_aniso)

        if max_aniso != 1:
            if not high_quality:
                uv_step_x = vt_dxdy_img[..., 0, :]
                uv_step_y = vt_dxdy_img[..., 1, :]

                with th.enable_grad():
                    uv_ext_x = th.cat(
                        [
                            grid + uv_step_x * ((j + 1) / (max_aniso + 1) * 2.0 - 1.0)
                            for j in range(max_aniso)
                        ],
                        dim=0,
                    )
                    uv_ext_y = th.cat(
                        [
                            grid + uv_step_y * ((j + 1) / (max_aniso + 1) * 2.0 - 1.0)
                            for j in range(max_aniso)
                        ],
                        dim=0,
                    )
                    uv_ext = th.where(
                        (px > py)[..., None].tile(max_aniso, 1, 1, 2),
                        uv_ext_x,
                        uv_ext_y,
                    )
            else:
                # From SVD we have direction of the maximum gradient in the uv space.
                # We ntegrate along this direction using `max_aniso` samples
                uv_step = (v[..., 0, :] * s[..., 0:1]) / size[None, None, None, :]
                with th.enable_grad():
                    uv_ext = th.cat(
                        [
                            grid + uv_step * ((j + 1) / (max_aniso + 1) * 2.0 - 1.0)
                            for j in range(max_aniso)
                        ],
                        dim=0,
                    )

    result = []
    if max_aniso == 1:
        for level in input:
            r = thf.grid_sample(
                level,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
            result.append(r)
    else:
        for level in input:
            r = thf.grid_sample(
                th.tile(level, (max_aniso, 1, 1, 1)),
                uv_ext,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
            r = r.view(max_aniso, r.shape[0] // max_aniso, *r.shape[1:]).mean(dim=0)
            result.append(r)
    return _combine_sampled_mipmaps(result, d1, a)


def _mipmap_selection(
    q: int,
    p_max: th.Tensor,
    p_min: Optional[th.Tensor],
    max_aniso: int = 1,
) -> Tuple[th.Tensor, th.Tensor]:
    if max_aniso != 1:
        # See p.255 of OpenGL Core Profile
        # N = min(ceil(Pmax/Pmin),maxAniso)
        N = th.clamp(th.ceil(p_max / p_min), max=max_aniso)
        N[th.isnan(N)] = 1

        # Lambda' = log2(Pmax/N)
        lambda_ = th.log2(p_max / N)
    else:
        lambda_ = th.log2(p_max)

    lambda_[th.isinf(lambda_)] = 0

    # See eq. 8.15, 8.16
    # Substract small number (1e-6) so that `lambda_` is always < q - 1
    lambda_ = th.clamp(lambda_, min=0, max=q - 1 - 1e-6)
    d1 = th.floor(lambda_).long()
    a = lambda_ - d1.float()
    return d1, a


def _compute_grad_magnitude(vt_dxdy_img: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    # See p.255 of OpenGL Core Profile
    # Px = sqrt(dudx^2 + dvdx^2)
    # Py = sqrt(dudy^2 + dvdy^2)
    px = th.norm(vt_dxdy_img[..., 0, :], dim=-1)
    py = th.norm(vt_dxdy_img[..., 1, :], dim=-1)
    return px, py


def _combine_sampled_mipmaps(
    sampled_mipmaps: List[th.Tensor], d1: th.Tensor, a: th.Tensor
) -> th.Tensor:
    if len(sampled_mipmaps) == 1:
        return sampled_mipmaps[0]
    sampled_mipmaps = th.stack(sampled_mipmaps, dim=0)
    indices = th.cat([d1[None, :, None], d1[None, :, None] + 1], dim=0)
    samples = th.gather(
        sampled_mipmaps,
        dim=0,
        index=indices.expand(-1, *sampled_mipmaps.shape[1:3], -1, -1),
    )
    # Interpolate two nearest mipmaps. See p.266
    return th.lerp(samples[0], samples[1], a[:, None])

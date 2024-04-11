from typing import List, Optional, Tuple

import torch as th
import torch.nn.functional as thf


def mipmap_grid_sample(
    input: List[th.Tensor],
    grid: th.Tensor,
    vt_dxdy_img: th.Tensor,
    max_aniso: int,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: Optional[bool] = False,
    high_quality: bool = False,
) -> th.Tensor:
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
        px, py = compute_grad_magnitude(vt_dxdy_img_pixel)

        if not high_quality:
            # This is what hardware implements.
            # We assume that maximum and minimum direction is either x or y.
            # This assumption is a quite grude approximation
            p_max = th.max(px, py)
            p_min = th.min(px, py) if max_aniso != 1 else None
        else:
            # A more correct version, which is not implemented in hardware
            # We find true principal axis
            u, s, v = th.linalg.svd(vt_dxdy_img_pixel)
            p_max = s[..., 0]
            p_min = s[..., 1]

        # Given the max and min gradients, select mipmap levels (assumes linear interpolation
        # between mipmaps)
        d1, a = mipmap_selection(q, p_max, p_min, max_aniso)

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
    return combine_sampled_mipmaps(result, d1, a)


def mipmap_selection(
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


def compute_grad_magnitude(vt_dxdy_img: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    # See p.255 of OpenGL Core Profile
    # Px = sqrt(dudx^2 + dvdx^2)
    # Py = sqrt(dudy^2 + dvdy^2)
    px = th.norm(vt_dxdy_img[..., 0, :], dim=-1)
    py = th.norm(vt_dxdy_img[..., 1, :], dim=-1)
    return px, py


def combine_sampled_mipmaps(
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

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch as th

DISTORTION_MODES: Set[Optional[str]] = {None, "radial-tangential", "fisheye"}


def project_pinhole(
    v_cam: th.Tensor, focal: th.Tensor, princpt: th.Tensor
) -> th.Tensor:
    """Project camera-space points to pixel-space points with camera
    intrinsics.

    v_cam:      N x V x 3
    focal:      N x 2 x 2
    princpt:    N x 2
    """

    z = v_cam[:, :, 2:3]
    z = th.where(z < 0, z.clamp(max=-1e-8), z.clamp(min=1e-8))

    v_proj = v_cam[:, :, 0:2] / z
    v_pix = (focal[:, None] @ v_proj[..., None])[..., 0] + princpt[:, None]

    return v_pix


def project_pinhole_distort_rt(
    v_cam: th.Tensor,
    focal: th.Tensor,
    princpt: th.Tensor,
    D: th.Tensor,
    fov: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Project camera-space points to distorted pixel-space using the radial
    and tangential model (4 parameters).

    v_cam:      N x V x 3
    focal:      N x 2 x 2
    princpt:    N x 2
    D:          N x 4
    fov:        N x 1
    """

    # See https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

    if fov is None:
        with th.no_grad():
            fov = estimate_rt_fov(D)

    z = v_cam[:, :, 2:3]
    z = th.where(z < 0, z.clamp(max=-1e-8), z.clamp(min=1e-8))

    v_proj = v_cam[:, :, :2] / z
    r2 = v_proj.pow(2).sum(-1)

    # Clamp x, y and r to avoid wrapping behavior of the distortion model.
    r2 = r2.clamp(max=fov.pow(2))
    v_clamped = v_proj.clamp(min=-fov[..., None], max=fov[..., None])

    assert D.shape[1] in [4, 5, 8]

    # 4 param: R = (1 + k1 r^2 + k2 r^4)
    R = 1 + D[:, 0:1] * r2 + D[:, 1:2] * r2.pow(2)

    # 5 param: R = (1 + k1 r^2 + k2 r^4 + k3 r^6)
    if D.shape[1] == 5:
        R = R + D[:, 4:5] * r2.pow(3)

    # 8 param: R = (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)
    if D.shape[1] == 8:
        R = R + D[:, 4:5] * r2.pow(3)
        R = R / (1 + D[:, 5:6] * r2 + D[:, 6:7] * r2.pow(2) + D[:, 7:8] * r2.pow(3))

    # [x' y'] * R
    v_proj_dist = v_proj * R[..., None]

    # [2 p1 x' y',  2 p2 x' y']
    v_proj_dist += (
        2
        * v_clamped[..., 0:1]
        * v_clamped[..., 1:2]
        * th.stack((D[:, 2:3], D[:, 3:4]), dim=-1)
    )
    # [p2 r^2,  p1 r^2]
    v_proj_dist += r2[..., None] * th.stack((D[:, 3:4], D[:, 2:3]), dim=-1)

    # [2 p2 x'^2, 2 p1 y'^2]
    v_proj_dist += th.stack(
        (
            2 * D[:, 3:4] * v_clamped[..., 0].pow(2),
            2 * D[:, 2:3] * v_clamped[..., 1].pow(2),
        ),
        dim=-1,
    )

    v_pix_dist = (focal[:, None] @ v_proj_dist[..., None])[..., 0] + princpt[:, None]

    return v_pix_dist


def project_fisheye_distort(
    v_cam: th.Tensor,
    focal: th.Tensor,
    princpt: th.Tensor,
    D: th.Tensor,
    fov: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Project camera-space points to distort pixel-space points using the
    fisheye distortion model.

    v_cam:      N x V x 3
    focal:      N x 2 x 2
    princpt:    N x 2
    D:          N x 4
    fov:        N x 1
    """

    # See https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fisheye.cpp

    if fov is None:
        with th.no_grad():
            fov = estimate_fisheye_fov(D)

    z = v_cam[:, :, 2:3]
    z = th.where(z < 0, z.clamp(max=-1e-8), z.clamp(min=1e-8))

    v_proj = v_cam[:, :, :2] / z
    r = v_proj.pow(2).sum(-1).sqrt()
    r = r.clamp(max=fov, min=1e-8 * th.ones_like(fov))
    theta = th.atan(r)
    theta_d = theta * (
        1
        + D[:, 0:1] * theta.pow(2)
        + D[:, 1:2] * theta.pow(4)
        + D[:, 2:3] * theta.pow(6)
        + D[:, 3:4] * theta.pow(8)
    )
    r = th.where(r < 0, r.clamp(max=-1e-8), r.clamp(min=1e-8))
    v_proj_dist = v_proj * (theta_d / r)[..., None]
    v_pix_dist = (focal[:, None] @ v_proj_dist[..., None])[..., 0] + princpt[:, None]

    return v_pix_dist


def project_fisheye_distort_62(
    v_cam: th.Tensor,
    focal: th.Tensor,
    princpt: th.Tensor,
    D: th.Tensor,
    fov: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Project camera-space points to distort pixel-space points using the
    OculusVisionFishEye62 distortion model.

    v_cam:      N x V x 3
    focal:      N x 2 x 2
    princpt:    N x 2
    D:          N x 4
    fov:        N x 1
    """

    # See https://www.internalfb.com/code/fbsource/[188bdaeaad64]/arvr/projects/nimble/prod/pynimble/visualization/shaders.py?lines=103-123
    # a more readible version: https://euratom-software.github.io/calcam/html/intro_theory.html
    if fov is None:
        with th.no_grad():
            fov = estimate_fisheye_fov(D)

    z = v_cam[:, :, 2:3]
    z = th.where(z < 0, z.clamp(max=-1e-8), z.clamp(min=1e-8))

    v_proj = v_cam[:, :, :2] / z
    r = v_proj.pow(2).sum(-1).sqrt()  # rp
    r = r.clamp(max=fov, min=1e-8 * th.ones_like(fov))
    theta = th.atan(r)
    theta_d = theta * (
        1
        + D[:, 0:1] * theta.pow(2)
        + D[:, 1:2] * theta.pow(4)
        + D[:, 2:3] * theta.pow(6)
        + D[:, 3:4] * theta.pow(8)
        + D[:, 4:5] * theta.pow(10)
        + D[:, 5:6] * theta.pow(12)
    )
    r = th.where(r < 0, r.clamp(max=-1e-8), r.clamp(min=1e-8))
    v_proj_dist = v_proj * (theta_d / r)[..., None]

    # Tangential Distortion
    x = v_proj_dist[:, :, 0]
    y = v_proj_dist[:, :, 1]

    xtan = D[:, 6:7] * (r.pow(2) + 2 * x.pow(2)) + 2 * D[:, 7:8] * x * y
    ytan = 2 * D[:, 6:7] * x * y + D[:, 7:8] * (r.pow(2) + 2 * y.pow(2))

    pTangential = th.cat([xtan[..., None], ytan[..., None]], dim=-1)

    v_proj_dist = v_proj_dist + pTangential

    v_pix_dist = (focal[:, None] @ v_proj_dist[..., None])[..., 0] + princpt[:, None]

    return v_pix_dist


def estimate_rt_fov(D: Union[np.ndarray, th.Tensor]) -> th.Tensor:
    """Estimate the maximum field of view based on the assumption that the 5th order
    polynomial for fish-eye effect is non-decreasing.

    D:  N x 4
    """

    if th.is_tensor(D):
        coefs = D.cpu().numpy()
    else:
        coefs = D

    ones = np.ones_like(coefs[:, 0])
    zeros = np.zeros_like(coefs[:, 0])
    coefs = np.stack(
        [
            5 * coefs[:, 1],
            zeros,
            3 * coefs[:, 0],
            zeros,
            ones,
        ],
        axis=-1,
    )

    fov = []
    for coef in coefs:
        roots = np.roots(coef)
        real_valued = roots.real[abs(roots.imag) < 1e-5]
        positive_roots = real_valued[real_valued > 0]
        if len(positive_roots) == 0:
            fov.append(np.inf)
        else:
            fov.append(positive_roots.min())
    fov = np.asarray(fov, dtype=np.float32)[..., None]

    if th.is_tensor(D):
        fov = th.from_numpy(fov).to(D)

    return fov


def estimate_fisheye_fov(D: Union[np.ndarray, th.Tensor]) -> th.Tensor:
    """Estimate the maximum field of view based on the assumption that the 9th order
    polynomial is non-decreasing.

    D:  N x 4
    """

    if th.is_tensor(D):
        coefs = D.cpu().numpy()
    else:
        coefs = D

    ones = np.ones_like(coefs[:, 0])
    zeros = np.zeros_like(coefs[:, 0])
    coefs = np.stack(
        [
            9 * coefs[:, -1],
            zeros,
            7 * coefs[:, -2],
            zeros,
            5 * coefs[:, -3],
            zeros,
            3 * coefs[:, -4],
            zeros,
            ones,
        ],
        axis=-1,
    )

    fov = []
    for coef in coefs:
        roots = np.roots(coef)
        real_valued = roots.real[abs(roots.imag) < 1e-5]
        positive_roots = real_valued[real_valued > 0]
        if len(positive_roots) == 0:
            fov.append(np.pi / 2)
        else:
            fov.append(min(positive_roots.min(), np.pi / 2))
    fov = np.asarray(np.tan(fov), dtype=np.float32)[..., None]

    if th.is_tensor(D):
        fov = th.from_numpy(fov).to(D)

    return fov


def project_points(
    v: th.Tensor,
    campos: th.Tensor,
    camrot: th.Tensor,
    focal: th.Tensor,
    princpt: th.Tensor,
    distortion_mode: Optional[Sequence[str]] = None,
    distortion_coeff: Optional[th.Tensor] = None,
    fov: Optional[th.Tensor] = None,
) -> Tuple[th.Tensor, th.Tensor]:
    """Project 3D world-space vertices to pixel-space, optionally applying a
    distortion model with provided coefficients.

    Returns v_pix, v_cam, both N x V x 3 since we preserve the camera-space
    Z-coordinate for v_pix.

    v:                  N x V x 3
    camrot:             N x 3 x 3
    campos:             N x 3
    focal:              N x 2 x 2
    princpt:            N x 2
    distortion_coeff:   N x 4
    fov:                N x 1
    """

    if distortion_mode is not None:
        assert distortion_coeff is not None, "Missing distortion coefficients."

    v_cam = (camrot[:, None] @ (v - campos[:, None])[..., None])[..., 0]

    # Fall back to single distortion mode if all the distortion modes are the same.
    if isinstance(distortion_mode, (list, tuple)):
        modes = list(set(distortion_mode))
        if len(modes) == 0:
            distortion_mode = None
        elif len(modes) == 1:
            distortion_mode = modes[0]

    if distortion_mode is None:
        v_pix = project_pinhole(v_cam, focal, princpt)
    elif isinstance(distortion_mode, str):
        assert distortion_coeff is not None

        # Single distortion model
        if distortion_mode == "radial-tangential":
            v_pix = project_pinhole_distort_rt(
                v_cam, focal, princpt, distortion_coeff, fov
            )
        elif distortion_mode == "fisheye":
            v_pix = project_fisheye_distort(
                v_cam, focal, princpt, distortion_coeff, fov
            )
        elif distortion_mode == "fisheye62":
            v_pix = project_fisheye_distort_62(
                v_cam, focal, princpt, distortion_coeff, fov
            )
        else:
            raise ValueError(
                f"Invalid distortion mode: {distortion_mode}. Valid options: {DISTORTION_MODES}."
            )
    elif isinstance(distortion_mode, (list, tuple)):
        assert distortion_coeff is not None

        # A mix of multiple distortion modes
        modes = set(distortion_mode)
        if not modes <= DISTORTION_MODES:
            raise ValueError(
                f"Invalid distortion mode: {distortion_mode}. Valid options: {DISTORTION_MODES}."
            )
        v_pix = th.empty_like(v_cam[..., :2])
        if None in modes:
            idx = th.tensor(
                [mode is None for mode in distortion_mode], device=v_pix.device
            )
            v_pix[idx] = project_pinhole(v_cam[idx], focal[idx], princpt[idx])
        if "radial-tangential" in modes:
            idx = th.tensor(
                [mode == "radial-tangential" for mode in distortion_mode],
                device=v_pix.device,
            )
            v_pix[idx] = project_pinhole_distort_rt(
                v_cam[idx],
                focal[idx],
                princpt[idx],
                distortion_coeff[idx],
                fov[idx] if fov is not None else None,
            )
        if "fisheye" in modes:
            idx = th.tensor(
                [mode == "fisheye" for mode in distortion_mode], device=v_pix.device
            )
            v_pix[idx] = project_fisheye_distort(
                v_cam[idx],
                focal[idx],
                princpt[idx],
                distortion_coeff[idx],
                fov[idx] if fov is not None else None,
            )
    else:
        raise ValueError(
            f"Invalid distortion mode: {distortion_mode}. Valid options: {DISTORTION_MODES}."
        )

    v_pix = th.cat((v_pix[:, :, 0:2], v_cam[:, :, 2:3]), dim=-1)

    return v_pix, v_cam


def project_points_grad(
    v_grad: th.Tensor,
    v: th.Tensor,
    campos: th.Tensor,
    camrot: th.Tensor,
    focal: th.Tensor,
    distortion_mode: Optional[Sequence[str]] = None,
    distortion_coeff: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Computes the gradient of projected (pixel-space) vertex positions with
    respect to the 3D world-space vertex positions given the gradient of the 3D
    world-space vertex positions.

    project_points_grad(dv, v) = d project_points(v) / dv * dv

    Args:
        v_grad: Gradient of 3D world-space vertices. Shape: N x V x 3
        v: 3D world-space vertices. Shape: N x V x 3
        camrot: Camera rotation. Shape:  N x 3 x 3
        camrot: Camera position. Shape:  N x 3
        focal: Focal length. Shape: N x 2 x 2
        distortion_mode: Distortion currently not implemented and must be None.
        distortion_coeff: Distortion currently not implemented and must be None.

    Returns:
         Gradient of 2D pixel-space vertices: N x V x 2
    """

    if distortion_mode is not None:
        assert distortion_coeff is not None, "Missing distortion coefficients."

    # d v_cam = d (Rv + T) = Rdv
    v_cam_grad = (camrot[:, None] @ v_grad[..., None])[..., 0]
    v_cam = (camrot[:, None] @ (v - campos[:, None])[..., None])[..., 0]

    if distortion_mode is None:
        z = v_cam[:, :, 2:3]
        z_grad = v_cam_grad[:, :, 2:3]
        z = th.where(z < 0, z.clamp(max=-1e-8), z.clamp(min=1e-8))

        # Using quotient rule:
        # d (v_cam / z) = (d v_cam * z - v_cam * dz) / z^2
        v_proj_grad = (v_cam_grad[:, :, 0:2] * z - v_cam[:, :, 0:2] * z_grad) / z**2.0

        # d v_pix = d (Kv + cp) = Kdv
        v_pix_grad = (focal[:, None] @ v_proj_grad[..., None])[..., 0]

    elif distortion_mode == "radial-tangential":
        raise NotImplementedError
    elif distortion_mode == "fisheye":
        raise NotImplementedError
    else:
        raise ValueError(
            f"Invalid distortion mode: {distortion_mode}. Valid options: {DISTORTION_MODES}."
        )

    return v_pix_grad

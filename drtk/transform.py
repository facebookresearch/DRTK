# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch as th
from drtk.utils import project_points


def transform(
    v: th.Tensor,
    campos: Optional[th.Tensor] = None,
    camrot: Optional[th.Tensor] = None,
    focal: Optional[th.Tensor] = None,
    princpt: Optional[th.Tensor] = None,
    K: Optional[th.Tensor] = None,
    Rt: Optional[th.Tensor] = None,
    distortion_mode: Optional[Union[List[str], str]] = None,
    distortion_coeff: Optional[th.Tensor] = None,
    fov: Optional[th.Tensor] = None,
) -> th.Tensor:
    """
    v: Tensor, N x V x 3
    Batch of vertex positions for vertices in the mesh.

    campos: Tensor, N x 3
    Camera position.

    camrot: Tensor, N x 3 x 3
    Camera rotation matrix.

    focal: Tensor, N x 2 x 2
    Focal length [[fx, 0],
                  [0, fy]]

    princpt: Tensor, N x 2
    Principal point [cx, cy]

    K: Tensor, N x 3 x 3
    Camera intrinsic calibration matrix. Either this or both (focal,
    princpt) must be provided.

    Rt: Tensor, N x 3 x 4 or N x 4 x 4
    Camera extrinsic matrix. Either this or both (camrot, campos) must be
    provided. Camrot is the upper 3x3 of Rt, campos = -R.T @ t.

    distortion_mode: List[str]
    Names of the distortion modes.

    distortion_coeff: Tensor, N x 4
    Distortion coefficients.

    fov: Tensor, N x 1
    Valid field of view of the distortion model.

    """

    v_pix, _ = transform_with_v_cam(
        v, campos, camrot, focal, princpt, K, Rt, distortion_mode, distortion_coeff, fov
    )

    return v_pix


def transform_with_v_cam(
    v: th.Tensor,
    campos: Optional[th.Tensor] = None,
    camrot: Optional[th.Tensor] = None,
    focal: Optional[th.Tensor] = None,
    princpt: Optional[th.Tensor] = None,
    K: Optional[th.Tensor] = None,
    Rt: Optional[th.Tensor] = None,
    distortion_mode: Optional[Union[List[str], str]] = None,
    distortion_coeff: Optional[th.Tensor] = None,
    fov: Optional[th.Tensor] = None,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Same as transform, but also returns the camera-space coordinates.
    In most cases it is not needed, but renderlayer depends on it
    """

    if not ((camrot is not None and campos is not None) ^ (Rt is not None)):
        raise ValueError("You must provide exactly one of Rt or (campos, camrot).")

    if not ((focal is not None and princpt is not None) ^ (K is not None)):
        raise ValueError("You must provide exactly one of K or (focal, princpt).")

    if campos is None:
        assert Rt is not None
        camrot = Rt[:, :3, :3]
        campos = -(camrot.transpose(-2, -1) @ Rt[:, :3, 3:4])[..., 0]

    if focal is None:
        assert K is not None
        focal = K[:, :2, :2]
        princpt = K[:, :2, 2]

    assert camrot is not None
    assert princpt is not None
    # Compute camera-space 3D coordinates and 2D pixel-space projections.
    v_pix, v_cam = project_points(
        v, campos, camrot, focal, princpt, distortion_mode, distortion_coeff, fov
    )

    return v_pix, v_cam

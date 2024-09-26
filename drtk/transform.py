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
    Projects 3D vertex positions onto the image plane of the camera.

    Args:
        v (th.Tensor):  vertex positions. N x V x 3
        campos (Tensor): Camera position. N x 3
        camrot (Tensor): Camera rotation matrix. N x 3 x 3
        focal (Tensor): Focal length. The upper left 2x2 block of the intrinsic matrix
            [[f_x, s], [0, f_y]].  N x 2 x 2
        princpt (Tensor): Camera principal point [cx, cy]. N x 2
        K (Tensor): Camera intrinsic calibration matrix, N x 3 x 3
        Rt (Tensor): Camera extrinsic matrix. N x 3 x 4 or N x 4 x 4
        distortion_mode (List[str]): Names of the distortion modes.
        distortion_coeff (Tensor): Distortion coefficients. N x 4
        fov (Tensor): Valid field of view of the distortion model. N x 1

    Returns:
        Vertex positions projected onto the image plane of the camera. The last dimension has
        still size 3. The first two components are the x and y coordinates on the image plane,
        and the z is z component of the vertex positions in the camera frame. The latter is used
        for depth values that are written to the z-buffer. N x V x 3

    .. warning::
        You must specify either ``K`` (intrinsic matrix) or both ``focal`` and ``princpt``
        (focal length and principal point).

        Additionally, you must provide either ``Rt`` (extrinsic matrix) or both ``campos``
        (camera position) and ``camrot`` (camera rotation).

    .. note::
        If we split ``Rt`` of shape N x 3 x 4 into ``R`` of shape N x 3 x 3 and ``t`` of
        shape N x 3 x 1, then: ``camrot`` is ``R``, and ``campos`` is ``-R.T @ t``.

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

from typing import Dict, Optional, Sequence, Tuple

import torch as th
import torch.nn.functional as thf
from drtk.interpolate import interpolate
from drtk.rasterize import rasterize
from drtk.render import render as _render
from drtk.renderlayer.projection import project_points


def transform(
    v: th.Tensor,
    campos: Optional[th.Tensor] = None,
    camrot: Optional[th.Tensor] = None,
    focal: Optional[th.Tensor] = None,
    princpt: Optional[th.Tensor] = None,
    K: Optional[th.Tensor] = None,
    Rt: Optional[th.Tensor] = None,
    distortion_mode: Optional[Sequence[str]] = None,
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

    if not ((camrot is not None and campos is not None) ^ (Rt is not None)):
        raise ValueError("You must provide exactly one of Rt or (campos, camrot).")

    if not ((focal is not None and princpt is not None) ^ (K is not None)):
        raise ValueError("You must provide exactly one of K or (focal, princpt).")

    if campos is None:
        assert Rt is not None
        camrot = Rt[:, :3, :3]
        campos = -(camrot.transpose(-2, -1) @ Rt[:, :3, 3:4])[..., 0]

    if focal is None:
        assert focal is not None
        focal = K[:, :2, :2]
        princpt = K[:, :2, 2]

    assert camrot is not None
    assert princpt is not None
    # Compute camera-space 3D coordinates and 2D pixel-space projections.
    v_pix, _ = project_points(
        v, campos, camrot, focal, princpt, distortion_mode, distortion_coeff, fov
    )

    return v_pix


def render(
    v: th.Tensor,
    vi: th.Tensor,
    vt: th.Tensor,
    vti: th.Tensor,
    tex: Optional[th.Tensor],
    size: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> Dict[str, th.Tensor]:

    if mode != "bilinear" and mode != "bicubic":
        raise ValueError(
            "rasterize(): only 'bilinear' and 'bicubic' modes are supported "
            "but got: '{}'".format(mode)
        )
    if (
        padding_mode != "zeros"
        and padding_mode != "border"
        and padding_mode != "reflection"
    ):
        raise ValueError(
            "rasterize(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            "but got: '{}'".format(padding_mode)
        )

    if align_corners is None:
        align_corners = False

    index_img = rasterize(v, vi, height=size[0], width=size[1])
    depth_img, bary_img = _render(v, vi, index_img)

    # In contrast to renderlayer API, the new modular API assumes that vt bas the batch dim.
    # Since the new kernels allow non-contiguous tensors, expanding the batch dim is free.
    # In order to work with both, vt with batch dim a nd without we need to expand it here
    # if it doesn't have a batch dim.
    if vt.ndim == 2:
        vt = vt[None].expand(v.shape[0], -1, -1)

    # The older render kernel scaled vt from 0..1 to -1..1 implicitly. Since we use a generic
    # `interpolate` function, we need to do scaling explicitly.
    vt_img = interpolate(2 * vt - 1.0, vti, index_img, bary_img).permute(0, 2, 3, 1)

    mask = th.ne(index_img, -1)

    out = {
        "depth_img": depth_img,
        "index_img": index_img,
        "vt_img": vt_img,
        "bary_img": bary_img,
        "mask": mask,
    }

    if tex is not None:
        res = thf.grid_sample(
            tex,
            vt_img,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        mf = mask[:, None].float()
        out["render"] = res * mf

    return out

import torch as th
from drtk import interpolate_ext

th.ops.load_library(interpolate_ext.__file__)


def interpolate(
    vert_attributes: th.Tensor,
    vi: th.Tensor,
    index_img: th.Tensor,
    bary_img: th.Tensor,
) -> th.Tensor:
    """
    Performs a linear interpolation of the vertex attributes given the barycentric coordinates
    Args:
        vert_attributes (th.Tensor):  vertex attribute tensor
            N x V x C
        vi (th.Tensor): face vertex index list tensor
            V x 3
        index_img (th.Tensor): index image tensor
            N x H x W
        bary_img (th.Tensor): 3D barycentric coordinate image tensor
            N x 3 x H x W
    Returns:
        A tensor with interpolated vertex attributes with a shape [N, C, H, W]
    Note:
        1. The default of `channels_last` is set to true to make this function backward compatible.
        Please consider using the argument `channels_last` instead of permuting the result afterward.
        2. By default, the output is not contiguous. Make sure you cal .contiguous() if that is a requirement.
    """
    return th.ops.interpolate_ext.interpolate(vert_attributes, vi, index_img, bary_img)


def interpolate_ref(
    vert_attributes: th.Tensor,
    vi: th.Tensor,
    index_img: th.Tensor,
    bary_img: th.Tensor,
) -> th.Tensor:
    """
    A reference implementation for `interpolate`. See the doc string from `interpolate`
    """
    b = vert_attributes.shape[0]
    iimg_clamped = index_img.clamp(min=0).long()
    vi_img = vi[iimg_clamped].long()

    v_img = th.gather(
        vert_attributes,
        1,
        vi_img.view(b, -1, 1).expand(-1, -1, vert_attributes.shape[-1]),
    )
    v_img = (
        v_img.view(*vi_img.shape[:3], 3, vert_attributes.shape[-1])
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    v_img = (v_img * bary_img[..., None]).sum(dim=1)
    return v_img.permute(0, 3, 1, 2)

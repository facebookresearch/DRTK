from typing import Callable, Optional

import torch as th
from drtk import edge_grad_ext
from drtk.interpolate import interpolate


th.ops.load_library(edge_grad_ext.__file__)


def edge_grad_estimator(
    v_pix: th.Tensor,
    vi: th.Tensor,
    bary_img: th.Tensor,
    img: th.Tensor,
    index_img: th.Tensor,
    v_pix_img_hook: Optional[Callable[[th.Tensor], None]] = None,
) -> th.Tensor:
    """
    Args:
        v_pix: Pixel-space vertex coordinates with preserved camera-space Z-values.
            N x V x 3

        vi: face vertex index list tensor
            V x 3

        bary_img: 3D barycentric coordinate image tensor
            N x 3 x H x W

        img: The rendered image
            N x C x H x W

        index_img: index image tensor
            N x H x W

        v_pix_img_hook: a backward hook that will be registered to v_pix_img. Useful for examining
            generated image space. Default None

    Returns:
        returns the img argument unchanged. Optionally also returns computed
        v_pix_img. Your loss should use the returned img, even though it is
        unchanged.

    Note:
        It is important to not spatially modify the rasterized image before passing it to edge_grad_estimator.
        Any operation as long as it is differentiable is ok after the edge_grad_estimator.

        Examples of opeartions that can be done before edge_grad_estimator:
            - Pixel-wise MLP
            - Color mapping
            - Color correction, gamma correction
        If the operation is significantly non-linear, then it is recommended to do it before
        edge_grad_estimator. All sorts of clipping and clamping (e.g. `x.clamp(min=0.0, max=1.0)`), must be
        done before edge_grad_estimator.

        Examples of operations that are not allowed before edge_grad_estimator:
            - Gaussian blur
            - Warping, deformation
            - Masking, cropping, making holes.

    Usage::

        from drtk.renderlayer import edge_grad_estimator

        ...

        out = renderlayer(v, tex, campos, camrot, focal, princpt,
                 output_filters=["index_img", "render", "mask", "bary_img", "v_pix"])

        img = out["render"] * out["mask"]

        img = edge_grad_estimator(
            v_pix=out["v_pix"],
            vi=rl.vi,
            bary_img=out["bary_img"],
            img=img,
            index_img=out["index_img"]
        )

        optim.zero_grad()
        image_loss = loss_func(img, img_gt)
        image_loss.backward()
        optim.step()
    """

    # Could use v_pix_img output from DRTK, but bary_img needs to be detached.
    v_pix_img = interpolate(v_pix, vi, index_img, bary_img.detach())

    img = th.ops.edge_grad_ext.edge_grad_estimator(v_pix, v_pix_img, vi, img, index_img)

    if v_pix_img_hook is not None:
        v_pix_img.register_hook(v_pix_img_hook)
    return img

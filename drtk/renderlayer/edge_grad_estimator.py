import torch as th
from drtk import edge_grad_ext
from drtk.interpolate import interpolate

th.ops.load_library(edge_grad_ext.__file__)


# pyre-fixme[3]: Return type must be annotated.
def edge_grad_estimator(
    v_pix: th.Tensor,
    vi: th.Tensor,
    bary_img: th.Tensor,
    img: th.Tensor,
    index_img: th.Tensor,
    return_v_pix_img: bool = False,
):
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

        return_v_pix_img: Bool - If True will return computed v_pix_img. Useful for adding backward
            pass hooks to visualize gradients. Default False

    Returns:
        returns the img argument unchanged. Optionally also returns computed
        v_pix_img. Your loss should use the returned img, even though it is
        unchanged. Gradients are computed in backward pass and are accumulated
        to the gradient of v_pix_img.

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

    # Temporary we switch to BHWC. Ater edge_grad kernel is updated this won't be necessary
    v_pix_img = v_pix_img.permute(0, 2, 3, 1).contiguous()

    with th.autocast(device_type="cuda", dtype=th.float32, enabled=False):
        img = th.ops.edge_grad_ext.edge_grad_estimator(
            v_pix, v_pix_img, vi, img, index_img
        )

    if return_v_pix_img:
        return img, v_pix_img
    return img

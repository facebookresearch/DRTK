# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as thf

from ..rasterizer import has_vulkan, rasterize, rasterize_packed

if has_vulkan:
    from ..rasterizer import VulkanRasterizerContext

from torch import Tensor
from torch.nn.modules.module import Module

from ..render_cuda import render as _render
from . import settings
from .geomutils import compute_vert_image, vert_binormals
from .projection import project_points, to_vulkan_clip_space
from .render_python import PythonRenderer
from .uv_grad import compute_uv_grad


class Mesh:
    def __init__(
        self,
        texcoords: th.Tensor,
        face_vert_inds: th.Tensor,
        face_texcoord_inds: th.Tensor,
    ) -> None:
        self.texcoords = texcoords
        self.face_vert_inds = face_vert_inds
        self.face_texcoord_inds = face_texcoord_inds

    @property
    def vt(self) -> Tensor:
        return self.texcoords

    @property
    def vi(self) -> Tensor:
        return self.face_vert_inds

    @property
    def vti(self) -> Tensor:
        return self.face_texcoord_inds


RENDERLAYER_OUTPUTS = {
    "render",
    "mask",
    "alpha",
    "vt_img",
    "bary_img",
    "vn_img",
    "vbn_img",
    "depth_img",
    "index_img",
    "v_pix_img",
    "v_cam_img",
    "v_img",
    "v_pix",
    "v_cam",
    "vt_dxdy_img",
}


class RenderLayer(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        vt: Union[np.ndarray, th.Tensor],
        vi: Union[np.ndarray, th.Tensor],
        vti: Union[np.ndarray, th.Tensor],
        boundary_aware: bool = False,
        flip_uvs: bool = True,
        grid_sample_params: Optional[Dict[str, Any]] = None,
        backface_culling: bool = False,
        use_vulkan: Optional[bool] = None,
        use_python_renderer: Optional[bool] = None,
    ) -> None:
        """Create a RenderLayer that produces w x h images."""
        super(RenderLayer, self).__init__()

        self.h = h
        self.w = w
        self.boundary_aware = boundary_aware
        self.flip_uvs = flip_uvs

        # pyre-fixme[4]: Attribute must be annotated.
        self.use_vulkan = use_vulkan if use_vulkan is not None else settings.use_vulkan
        # pyre-fixme[4]: Attribute must be annotated.
        self.use_python_renderer = (
            use_python_renderer
            if use_python_renderer is not None
            else settings.use_python_renderer
        )

        if self.use_vulkan:
            assert has_vulkan

        if self.use_python_renderer:
            # pyre-fixme[4]: Attribute must be annotated.
            self.pyrender = PythonRenderer(h, w)

        # pyre-fixme[4]: Attribute must be annotated.
        self.grid_sample_params = grid_sample_params or {
            "mode": "bilinear",
            "align_corners": False,
        }
        self.backface_culling = backface_culling

        # This is particularly important for rendering differentiable mask
        # losses where we use textures filled with solid colors. On the border,
        # if we interpolate with zeros outside, we get a vastly different
        # result for some texels.
        self.grid_sample_params["padding_mode"] = "border"

        if not isinstance(vt, th.Tensor):
            vt = th.from_numpy(vt)
        if not isinstance(vi, th.Tensor):
            vi = th.from_numpy(vi)
        if not isinstance(vti, th.Tensor):
            vti = th.from_numpy(vti)

        self.register_buffer("vt", vt.clone().float().contiguous(), persistent=False)
        self.register_buffer("vi", vi.clone().int().contiguous(), persistent=False)
        self.register_buffer("vti", vti.clone().int().contiguous(), persistent=False)

        if flip_uvs:
            self.vt[:, 1] = 1 - self.vt[:, 1]

        # The Vulkan rasterizer context needs to start with a CUDA device of
        # some kind so that it can choose the matching Vulkan physical device.
        # We change this appropriately in our overridden to() method.
        if self.use_vulkan:
            init_vrc_dev = th.cuda.current_device()
            # pyre-fixme[4]: Attribute must be annotated.
            self.vrc = VulkanRasterizerContext(h, w, init_vrc_dev, backface_culling)
            self.vrc.updateTopology(self.vi.to(init_vrc_dev))

        if boundary_aware:
            k = th.zeros(2, 1, 3, 3)
            k[0, 0] = th.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
            k[1, 0] = th.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4
            self.register_buffer("sobel_kernel", k, persistent=False)

    # pyre-fixme[2]: Parameter must be annotated.
    def to(self, *args, **kwargs) -> Module:
        device = th._C._nn._parse_to(*args, **kwargs)[0]
        if device.type != "cuda":
            raise ValueError(f"Device {device} is not a CUDA device.")
        idx = device.index
        if self.use_vulkan:
            self.vrc.to(idx if idx is not None else th.cuda.current_device())
        return super().to(*args, **kwargs)

    def resize(self, h: int, w: int) -> None:
        self.h = h
        self.w = w

    # pyre-fixme[3]: Return type must be annotated.
    def rasterize(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        v,
        # pyre-fixme[2]: Parameter must be annotated.
        v_clip,
        # pyre-fixme[2]: Parameter must be annotated.
        mesh,
        # pyre-fixme[2]: Parameter must be annotated.
        campos,
        # pyre-fixme[2]: Parameter must be annotated.
        camrot,
        # pyre-fixme[2]: Parameter must be annotated.
        focal,
        # pyre-fixme[2]: Parameter must be annotated.
        princpt,
        write_depth: bool = False,
    ):
        assert v.dtype == th.float32
        assert v_clip.dtype == th.float32
        vi = mesh.vi

        dev = v.device
        bs = v.shape[0]
        depth_img = th.empty(bs, self.h, self.w, dtype=th.float32, device=dev)
        index_img = th.empty(bs, self.h, self.w, dtype=th.int32, device=dev)

        if self.use_vulkan:
            if vi is not self.vi:
                self.vrc.updateTopology(vi)
            return rasterize(
                self.vrc, v_clip.contiguous(), depth_img, index_img, write_depth
            )
        packed_index_img = th.empty(bs, self.h, self.w, 2, dtype=th.int32, device=dev)
        return rasterize_packed(
            v_clip.contiguous(), vi, depth_img, index_img, packed_index_img
        )[:2]

    # pyre-fixme[2]: Parameter must be annotated.
    def interp_vert_attrs(self, v_pix, mesh, index_img, vn=None) -> Dict[str, Any]:
        assert index_img.dtype == th.int
        vt = mesh.vt
        vi = mesh.vi
        vti = mesh.vti

        if self.use_python_renderer:
            depth_img, bary_img, vt_img, vn_img = self.pyrender(
                v_pix, vt, vi, vti, index_img, vn
            )
        else:
            _render_outs = _render(v_pix, vt, vi, vti, index_img, vn)
            depth_img, bary_img, vt_img = _render_outs[:3]
            if vn is not None:
                vn_img = _render_outs[3]

        out = {
            "depth_img": depth_img,
            "vt_img": vt_img,
            "bary_img": bary_img.permute(0, 3, 1, 2),
        }
        if vn is not None:
            # pyre-fixme[61]: `vn_img` is undefined, or not always defined.
            out["vn_img"] = vn_img
        return out

    def render_ba(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        v_pix,
        tex: Tensor,
        # pyre-fixme[2]: Parameter must be annotated.
        mesh,
        index_img: Tensor,
        # pyre-fixme[2]: Parameter must be annotated.
        mask,
        # pyre-fixme[2]: Parameter must be annotated.
        vn,
        # pyre-fixme[2]: Parameter must be annotated.
        background,
        # pyre-fixme[2]: Parameter must be annotated.
        ksize,
    ) -> Dict[str, Any]:
        if (ksize % 2) == 0 or ksize < 3:
            raise ValueError(
                "Invalid kernel size for boundary-aware blurring. ksize must be an odd integer >= 3."
            )

        krad = ksize // 2
        b = v_pix.shape[0]

        weighted_img_sum = th.zeros(
            b, tex.shape[1], self.h, self.w, device=v_pix.device
        )
        weight_sum = th.zeros(b, 1, self.h, self.w, device=v_pix.device)
        masked_weight_sum = th.zeros(b, 1, self.h, self.w, device=v_pix.device)
        if vn is not None:
            weighted_vn_sum = th.zeros((b, self.h, self.w, 3), device=v_pix.device)

        weight_sum.fill_(0)
        masked_weight_sum.fill_(0)
        weighted_img_sum.fill_(0)

        # For pixels that do not lie near a depth discontinuity, use the
        # plain rendering.
        base_render_out = self.interp_vert_attrs(v_pix, mesh, index_img, vn=vn)
        base_vt_img = base_render_out["vt_img"]
        base_render = thf.grid_sample(tex, base_vt_img, **self.grid_sample_params)
        float_mask = mask[:, None].float()
        base_render = base_render * float_mask

        # pyre-fixme[6]: Expected `List[int]` for 2nd param but got
        #  `Tuple[typing.Any, typing.Any, typing.Any, typing.Any]`.
        index_img_padded = thf.pad(index_img, (krad, krad, krad, krad), value=-2)

        # Blur contributions from pixels within a neighborhood of a depth
        # discontinuity so that we spread gradients w.r.t. vertex positions
        # across discontinuity boundaries.
        h = self.h
        w = self.w
        for y_off in range(ksize):
            for x_off in range(ksize):
                if x_off == krad and y_off == krad:
                    # Don't re-render the center pixel, just use the base output.
                    render_out = base_render_out
                    iimg_shifted = index_img
                    mask_shifted = float_mask
                else:
                    iimg_shifted = index_img_padded[
                        :, y_off : y_off + h, x_off : x_off + w
                    ]
                    iimg_shifted = iimg_shifted.contiguous()
                    mask_shifted = (iimg_shifted > -1)[:, None].float()

                    render_out = self.interp_vert_attrs(
                        v_pix, mesh, iimg_shifted, vn=vn
                    )

                vt_img = render_out["vt_img"]
                bary_img = render_out["bary_img"]

                # Don't count pixels outside the border as background pixels
                # since we don't know if there is background or more triangles
                # outside the image bounds.
                non_border = (iimg_shifted != -2)[:, None].float()

                weight = 1 / (-bary_img.clamp(max=0).sum(dim=1, keepdim=True) * 5 + 1)
                render = thf.grid_sample(tex, vt_img, **self.grid_sample_params)

                masked_weight_sum = masked_weight_sum + weight * mask_shifted
                weighted_img_sum = weighted_img_sum + render * mask_shifted * weight
                weight_sum = weight_sum + weight * non_border

                if vn is not None:
                    normals = render_out["vn_img"]
                    # pyre-fixme[16]: `float` has no attribute `permute`.
                    wp = weight.permute(0, 2, 3, 1)
                    mp = mask_shifted.permute(0, 2, 3, 1)
                    weighted_vn_sum = weighted_vn_sum + normals * wp * mp

        alpha = masked_weight_sum / weight_sum.clamp(min=1e-8)
        render = weighted_img_sum / masked_weight_sum.clamp(min=1e-8)

        # For blurring normals, it doesn't make sense to blend them with the
        # background since the background is color, not normals. Instead, just
        # blend across meshes and re-normalize.
        if vn is not None:
            # pyre-fixme[61]: `weighted_vn_sum` is undefined, or not always defined.
            normals = thf.normalize(weighted_vn_sum, dim=-1)
            normals = normals * float_mask.permute(0, 2, 3, 1)

        # Any pixels that have one or more non-background pixels mixed into
        # them (alpha > 0) are now considered valid.
        mask = (alpha > 0)[:, 0]

        render = alpha * render
        if background is not None:
            render = render + (1 - alpha) * background

        out = {
            "vt_img": base_vt_img,
            "render": render,
            "alpha": alpha,
            "mask": mask,
            "depth_img": base_render_out["depth_img"],
            "bary_img": base_render_out["bary_img"],
        }

        if vn is not None:
            # pyre-fixme[61]: `normals` is undefined, or not always defined.
            out["vn_img"] = normals

        return out

    # TODO: Should we finally move to having people pass in a "camera
    # parameters" struct that holds / computes the KRt matrix?
    def forward(
        self,
        v: th.Tensor,
        tex: th.Tensor,
        campos: Optional[th.Tensor] = None,
        camrot: Optional[th.Tensor] = None,
        focal: Optional[th.Tensor] = None,
        princpt: Optional[th.Tensor] = None,
        K: Optional[th.Tensor] = None,
        Rt: Optional[th.Tensor] = None,
        vt: Optional[th.Tensor] = None,
        vi: Optional[th.Tensor] = None,
        vti: Optional[th.Tensor] = None,
        distortion_mode: Optional[Sequence[str]] = None,
        distortion_coeff: Optional[th.Tensor] = None,
        fov: Optional[th.Tensor] = None,
        vn: Optional[th.Tensor] = None,
        depth_img: Optional[th.Tensor] = None,
        background: Optional[th.Tensor] = None,
        ksize: Optional[int] = None,
        far: Optional[float] = None,
        near: Optional[float] = None,
        output_filters: Optional[Sequence[str]] = None,
    ) -> Dict[str, th.Tensor]:
        """
        v: Tensor, N x V x 3
        Batch of vertex positions for vertices in the mesh.

        tex: Tensor, N x C x H x W
        Batch of textures to render on the mesh.

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

        vt: Tensor, Ntexcoords x 2
        Optional texcoords to use. If given, they override the ones
        used to construct this RenderLayer.

        vi: Tensor, Nfaces x 3
        Optional face vertex indices to use. If given, they override the ones
        used to construct this RenderLayer.

        vti: Tensor, Nfaces x 3
        Optional face texcoord indices to use. If given, they override the ones
        used to construct this RenderLayer.

        distortion_mode: Sequence[str]
        Names of the distortion modes.

        distortion_coeff: Tensor, N x 4
        Distortion coefficients.

        fov: Tensor, N x 1
        Valid field of view of the distortion model.

        depth_img: Tensor, N x H x W
        Optional pre-existing depth map. Render triangles on top of this depth
        map, discarding any triangles that lie behind the surface represented
        by the map.

        vn: Tensor, N x Nverts x 3
        Optional vertex normals. If given, they will be interpolated along the
        surface to give per-pixel interpolated normals.

        background: Tensor, N x C x H x W
        Background images on which to composite the rendered mesh.

        far: float
        Far plane.

        near: float
        Near plane.

        output_filters: Sequence[str]
        List of output names to return. Not returning unused outputs can save GPU
        memory. Valid output names:

        render:     The rendered masked image.
                    N x C x H x W

        mask:       Mask of which pixels contain valid rendered colors.
                    N x H x W

        alpha:      Soft foreground / background mask (if boundary_aware is on and ksize > 1).
                    N x 1 x H x W

        vt_img:     Per-pixel interpolated texture coordinates.
                    N x H x W x 2

        bary_img:   Per-pixel interpolated 3D barycentric coordinates.
                    N x 3 x H x W

        vn_img:     Per-pixel interpolated vertex normals (if vn was given).
                    N x H x W x 3

        vbn_img:    Per-pixel interpolated vertex binormals (if vn was given).
                    N x H x W x 3

        depth_img:  Per-pixel depth values.
                    N x H x W

        index_img:  Per-pixel face indices.
                    N x H x W

        v_pix_img:  Per-pixel pixel-space vertex coordinates with preserved camera-space Z-values.
                    N x H x W x 3

        v_cam_img:  Per-pixel camera-space vertex coordinates.
                    N x H x W x 3

        v_img:      Per-pixel vertex coordinates.
                    N x H x W x 3

        v_pix:      Pixel-space vertex coordinates with preserved camera-space Z-values.
                    N x V x 3

        v_cam:      Camera-space vertex coordinates.
                    N x V x 3

        vt_dxdy_img: Per-pixel uv gradients with respect to the pixel-space position.
                     vt_dxdy_img is transposed Jacobian: (dt / dp_pix)^T, where:
                        t - uv coordinates, p_pix - pixel-space coordinates
                        vt_dxdy_img[..., i, j] = dt[j] / dp_pix[i]
                     e.i. image of 2x2 Jacobian matrices of the form: [[du/dx, dv/dx],
                                                                       [du/dy, dv/dy]]
                    N x H x W x 2 x 2

        all:        All of the above.
        """
        if output_filters is None:
            output_filters = ["render"]

        vt = vt if vt is not None else self.vt
        vi = vi if vi is not None else self.vi
        vti = vti if vti is not None else self.vti
        assert vti.shape[-2] == vi.shape[-2]
        mesh = Mesh(vt, vi, vti)

        if "all" in output_filters or (
            isinstance(output_filters, str) and output_filters == "all"
        ):
            output_filters = list(RENDERLAYER_OUTPUTS)

        unknown_filters = [f for f in output_filters if f not in RENDERLAYER_OUTPUTS]
        if len(unknown_filters) > 0:
            raise ValueError(
                "RenderLayer does not produce these outputs:", ",".join(unknown_filters)
            )

        if ksize is not None and not self.boundary_aware:
            raise ValueError(
                "You must initialize RenderLayer with `boundary_aware=True` to use boundary aware rendering (ksize != None)"
            )

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

        if Rt is None:
            Rt = (
                th.eye(4, dtype=v.dtype, device=v.device)[None]
                .expand(v.shape[0], -1, -1)
                .clone()
            )
            Rt[:, :3, :3] = camrot
            Rt[:, :3, 3] = -(camrot @ campos[..., None])[..., 0]
        elif Rt.shape[1] == 3:
            _Rt = (
                th.eye(4, dtype=v.dtype, device=v.device)[None]
                .expand(v.shape[0], -1, -1)
                .clone()
            )
            _Rt[:, :3, :4] = Rt
            Rt = _Rt

        # Compute camera-space 3D coordinates and 2D pixel-space projections.
        v_pix, v_cam = project_points(
            v, campos, camrot, focal, princpt, distortion_mode, distortion_coeff, fov
        )

        with th.no_grad():
            if self.use_vulkan:
                v_clip, A, B = to_vulkan_clip_space(
                    v_pix, self.h, self.w, far=far, near=near
                )
                _depth_img, index_img = self.rasterize(
                    v,
                    v_clip,
                    mesh,
                    campos,
                    camrot,
                    focal,
                    princpt,
                    write_depth=depth_img is not None,
                )
            else:
                if near is not None or far is not None:
                    raise NotImplementedError(
                        "Clipping planes are currently only supported by the Vulkan rasterizer."
                    )
                _depth_img, index_img = self.rasterize(
                    v,
                    v_pix,
                    mesh,
                    campos,
                    camrot,
                    focal,
                    princpt,
                )

            mask = th.ne(index_img, -1)

            if depth_img is not None:
                # Based on the projection matrix above, the depth buffer will
                # contain (Az + B) / w = A + B / z which we need to solve to
                # recover depth:
                #
                #   depth = A + B / z
                #   z = B / (depth - A)
                #
                # NOTE: this loses some precision compared to adding an
                # additional attachment for linear depth output, but it saves
                # memory and compute. Given that we use a 32-bit float depth
                # buffer, this should be ok for most uses.
                if self.use_vulkan:
                    _depth_img = (
                        mask.float()
                        * B[:, None, None]
                        / ((1 - _depth_img) - A[:, None, None])
                    )
                index_img[(_depth_img >= depth_img)] = -1

        if self.boundary_aware and ksize is not None:
            render_out = self.render_ba(
                v_pix, tex, mesh, index_img, mask, vn, background, ksize
            )
        else:
            render_out = self.interp_vert_attrs(v_pix, mesh, index_img, vn)
            render_out["mask"] = mask

            if "render" in output_filters:
                vt_img = render_out["vt_img"]
                render = thf.grid_sample(tex, vt_img, **self.grid_sample_params)

                mf = mask[:, None].float()
                render = render * mf
                if background is not None:
                    render = render + (1 - mf) * background

                render_out["render"] = render

        render_out["v_pix"] = v_pix
        render_out["v_cam"] = v_cam
        render_out["index_img"] = index_img
        bary_img = render_out["bary_img"]

        if "v_pix_img" in output_filters:
            render_out["v_pix_img"] = compute_vert_image(
                v_pix, mesh.vi, index_img, bary_img
            )

        if "v_cam_img" in output_filters:
            render_out["v_cam_img"] = compute_vert_image(
                v_cam, mesh.vi, index_img, bary_img
            )

        if "v_img" in output_filters:
            render_out["v_img"] = compute_vert_image(v, mesh.vi, index_img, bary_img)

        if "vbn_img" in output_filters:
            vbnorms = vert_binormals(v, vt, vi.long(), vti.long())
            render_out["vbn_img"] = compute_vert_image(
                vbnorms, mesh.vi, index_img, bary_img
            )

        if "vt_dxdy_img" in output_filters:
            render_out["vt_dxdy_img"] = compute_uv_grad(
                v,
                vt,
                vi,
                vti,
                index_img,
                bary_img,
                mask,
                campos,
                camrot,
                focal,
                distortion_mode,
                distortion_coeff,
            )

        return {k: v for k, v in render_out.items() if k in output_filters}

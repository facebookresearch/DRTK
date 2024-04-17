# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import torch as th
from care.strict.data.io.typed.file.obj_np import ObjFileNumpy
from care.strict.utils.geom import vert_normals
from rpack import enclosing_size, pack
from torch import Tensor

from .renderlayer import RenderLayer


TexBoundsType = List[Tuple[Tuple[int, int], Tuple[int, int]]]
AtlasSizeType = Tuple[int, int]


def compute_packed_uvs(
    tex_sizes: List[Tuple[int, int]], vts: List[th.Tensor], flip_uvs: bool = True
) -> Tuple[List[th.Tensor], List[Tuple[int, int]], TexBoundsType, AtlasSizeType]:
    """Pack a set of UV spaces into a single atlas, applying the appropriate
    scale/shift s.t. the UVs map properly into the new space.

    Args:
        tex_sizes: List of texture shapes, (H, W) for each texture.

        vts: List of UV coordinate tensors, one for each texture.

        flip_uvs: Indicates whether the UVs will be filpped vertically by some
            future consumer. They will not be flipped here, but the packing
            process will take this into account.

    Returns: (new_vts, tex_pos, tex_bounds, atlas_size)
        new_vts: A list of transformed UV coords, one per texture.
        tex_pos: Top-left corners of each texture in the atlas.
        tex_bounds: ((y0, y1), (x0, x1)) for each texture in the atlas.
        atlas_size: (H, W) the size of the packed texture atlas.
    """

    tex_positions = pack(tex_sizes)
    atlas_size = enclosing_size(tex_sizes, tex_positions)
    tex_bounds = [
        ((y, y + th), (x, x + tw)) for (th, tw), (y, x) in zip(tex_sizes, tex_positions)
    ]

    # Compute and apply UV offsets (scale + translation).
    uv_scales = [(tw / atlas_size[1], th / atlas_size[0]) for th, tw in tex_sizes]
    if flip_uvs:
        uv_ofs = [
            (x0 / atlas_size[1], (atlas_size[0] - (y0 + th)) / atlas_size[0])
            for (y0, x0), (th, tw) in zip(tex_positions, tex_sizes)
        ]
    else:
        uv_ofs = [
            (x0 / atlas_size[1], y0 / atlas_size[0]) for (y0, x0) in tex_positions
        ]

    new_vts = []
    for vt, ofs, scale in zip(vts, uv_ofs, uv_scales):
        ofs = (ofs[0], ofs[1])
        vt = (
            vt * th.tensor(scale, device=vt.device, dtype=vt.dtype)[None]
            + th.tensor(ofs, device=vt.device, dtype=vt.dtype)[None]
        )
        new_vts.append(vt)

    return new_vts, tex_positions, tex_bounds, atlas_size


def merge_meshes(
    n_verts: List[int],
    vts: List[th.Tensor],
    vis: List[th.Tensor],
    vtis: List[th.Tensor],
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, List[Tuple[int, int]]]:
    n_texcoords = [vt.shape[0] for vt in vts]
    n_prims = [vi.shape[0] for vi in vis]

    all_vt = th.cat(vts, dim=0)
    all_vi = th.cat(vis, dim=0)
    all_vti = th.cat(vtis, dim=0)

    cur_prim_ofs = n_prims[0]
    cur_vert_ofs = n_verts[0]
    cur_texcoord_ofs = n_texcoords[0]
    prim_idx_ranges = [(0, n_prims[0])]
    for n_vert, n_texcoord, n_prim in zip(n_verts[1:], n_texcoords[1:], n_prims[1:]):
        all_vi[cur_prim_ofs : cur_prim_ofs + n_prim] += cur_vert_ofs
        all_vti[cur_prim_ofs : cur_prim_ofs + n_prim] += cur_texcoord_ofs
        prim_idx_ranges.append((cur_prim_ofs, cur_prim_ofs + n_prim))

        cur_prim_ofs += n_prim
        cur_vert_ofs += n_vert
        cur_texcoord_ofs += n_texcoord
    return all_vt, all_vi, all_vti, prim_idx_ranges


def pack_texs(
    texs: List[Tensor],
    out: Tensor,
    tex_bounds: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> th.Tensor:
    for i, ((y0, y1), (x0, x1)) in enumerate(tex_bounds):
        out[:, : texs[i].shape[1], y0:y1, x0:x1] = texs[i]
    return out


# pyre-fixme[3]: Return type must be annotated.
def unpack_texs(tex: Tensor, tex_bounds: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    return [tex[:, :, tb[0][0] : tb[0][1], tb[1][0] : tb[1][1]] for tb in tex_bounds]


class MultiMeshRenderLayer(th.nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        n_verts: List[int],
        vts: Union[List[np.ndarray], List[th.Tensor]],
        vis: Union[List[np.ndarray], List[th.Tensor]],
        vtis: Union[List[np.ndarray], List[th.Tensor]],
        tex_sizes: List[Tuple[int, int]],
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        super().__init__()
        assert [vi.shape[0] == vti.shape[0] for vi, vti in zip(vis, vtis)]
        assert len(tex_sizes) == len(vts)

        vts: List[th.Tensor] = [
            vt if isinstance(vt, th.Tensor) else th.from_numpy(vt) for vt in vts
        ]
        vis: List[th.Tensor] = [
            vi if isinstance(vi, th.Tensor) else th.from_numpy(vi) for vi in vis
        ]
        vtis: List[th.Tensor] = [
            vti if isinstance(vti, th.Tensor) else th.from_numpy(vti) for vti in vtis
        ]

        #
        # Pack the textures into an atlas. This is required for boundary-aware
        # rendering.
        #
        flip_uvs = kwargs.get("flip_uvs", True)
        vts, tex_positions, tex_bounds, atlas_size = compute_packed_uvs(
            tex_sizes, vts, flip_uvs
        )

        self.tex_sizes = tex_sizes
        # pyre-fixme[4]: Attribute must be annotated.
        self.tex_positions = tex_positions
        # pyre-fixme[4]: Attribute must be annotated.
        self.tex_bounds = tex_bounds
        # pyre-fixme[4]: Attribute must be annotated.
        self.atlas_size = atlas_size

        #
        # Merge the index arrays and UVs into monolithic arrays for the
        # combined mesh.
        #
        all_vt, all_vi, all_vti, prim_idx_ranges = merge_meshes(n_verts, vts, vis, vtis)
        # pyre-fixme[4]: Attribute must be annotated.
        self.packed_vt = all_vt
        # pyre-fixme[4]: Attribute must be annotated.
        self.packed_vi = all_vi
        # pyre-fixme[4]: Attribute must be annotated.
        self.packed_vti = all_vti
        # pyre-fixme[4]: Attribute must be annotated.
        self.prim_idx_ranges = prim_idx_ranges

        self.rl = RenderLayer(h, w, all_vt, all_vi, all_vti, **kwargs)

    def resize(self, h: int, w: int) -> None:
        self.rl.resize(h, w)

    @property
    def vt(self) -> Tensor:
        return self.rl.vt

    @property
    def vi(self) -> Tensor:
        return self.rl.vi

    @property
    def vti(self) -> Tensor:
        return self.rl.vti

    def pack_texs(self, texs: List[th.Tensor]) -> th.Tensor:
        bs = texs[0].shape[0]
        max_chans = max(tex.shape[1] for tex in texs)
        dev = texs[0].device
        packed_tex_buf = th.zeros(
            bs, max_chans, self.atlas_size[0], self.atlas_size[1], device=dev
        )
        return pack_texs(texs, packed_tex_buf, self.tex_bounds)

    # pyre-fixme[3]: Return type must be annotated.
    def unpack_texs(self, tex: Tensor):
        return unpack_texs(tex, self.tex_bounds)

    def forward(
        self,
        verts: List[th.Tensor],
        texs: List[th.Tensor],
        campos: th.Tensor,
        camrot: th.Tensor,
        focal: th.Tensor,
        princpt: th.Tensor,
        output_filters: Optional[List[str]] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        """Renders a list of meshes with a list of textures. See RenderLayer for further documentation.

        This RenderLayer adds the following output filters:

        segments:   An index mask identifying which mesh was rendered to each pixel.
                    N x H x W
        """
        if output_filters is None:
            output_filters = ["render"]

        return_segments = "segments" in output_filters
        render_vn = "vn_img" in output_filters
        keep_mask = "mask" in output_filters
        keep_index_img = "index_img" in output_filters

        if return_segments:
            output_filters.remove("segments")
            if not keep_mask:
                output_filters.append("mask")
            if not keep_index_img:
                output_filters.append("index_img")

        v = th.cat(verts, dim=1)
        packed_tex = self.pack_texs(texs) if texs is not None else None

        vn = None
        if render_vn:
            vn = (
                camrot[:, None] @ vert_normals(v, self.rl.vi[None].long())[..., None]
            )[..., 0]

        out = self.rl(
            v=v,
            tex=packed_tex,
            campos=campos,
            camrot=camrot,
            focal=focal,
            princpt=princpt,
            output_filters=output_filters,
            vn=vn,
            **kwargs,
        )

        if return_segments:
            mask = out["mask"]
            index_img = out["index_img"]

            out["segments"] = [
                ((index_img >= rng[0]) & (index_img < rng[1]) & mask)[:, None]
                for rng in self.prim_idx_ranges
            ]

            if not keep_mask:
                del out["mask"]
            if not keep_index_img:
                del out["index_img"]

        return out


def make_multimesh_from_objs(
    filenames: List[str],
    h: int,
    w: int,
    tex_sizes: List[Tuple[int, int]],
    extra_meshes: Optional[
        List[Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]
    ] = None,
    # pyre-fixme[2]: Parameter must be annotated.
    **kwargs,
) -> MultiMeshRenderLayer:
    if extra_meshes is None:
        extra_meshes = []

    objs = [ObjFileNumpy.load(fn) for fn in filenames]
    vs: List[th.Tensor] = [th.from_numpy(obj["v"][:, :3]).float() for obj in objs]
    vts: List[th.Tensor] = [th.from_numpy(obj["vt"][:, :2]).float() for obj in objs]
    vis: List[th.Tensor] = [th.from_numpy(obj["vi"][:, :3]).int() for obj in objs]
    vtis: List[th.Tensor] = [th.from_numpy(obj["vti"][:, :3]).int() for obj in objs]
    n_verts = [v.shape[0] for v in vs]

    for extra_v, extra_vt, extra_vi, extra_vti in extra_meshes:
        if isinstance(extra_v, int):
            n_verts.append(extra_v)
        else:
            n_verts.append(extra_v.shape[0])
        vts.append(extra_vt.float())
        vis.append(extra_vi.int())
        vtis.append(extra_vti.int())

    return MultiMeshRenderLayer(
        h, w, n_verts, vts, vis, vtis, tex_sizes=tex_sizes, **kwargs
    )

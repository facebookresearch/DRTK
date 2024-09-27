# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch as th
from drtk.utils import load_torch_ops

load_torch_ops("drtk.rasterize_ext")


@th.compiler.disable
def rasterize(
    v: th.Tensor,
    vi: th.Tensor,
    height: int,
    width: int,
    wireframe: bool = False,
) -> th.Tensor:
    """
    Rasterizes a mesh defined by v and vi.

    Args:
        v (th.Tensor):  vertex positions. The first two components are the projected vertex's
            location (x, y) on the image plane. The coordinates of the top left corner are
            (-0.5, -0.5), and the coordinates of the bottom right corner are
            (width - 0.5, height - 0.5). The z component is expected to be in the camera space
            coordinate frame (before projection).
            N x V x 3

        vi (th.Tensor): face vertex index list tensor. The most significant nibble of vi is
            reserved for controlling visibility of the edges in wireframe mode. In non-wireframe
            mode, content of the most significant nibble of vi will be ignored.
            V x 3

        height (int): height of the image in pixels.

        width (int): width of the image in pixels.

        wireframe (bool): If False (default), rasterizes triangles. If True, rasterizes lines,
            where the most significant nibble of vi is reinterpreted as a bit field controlling
            the visibility of the edges. The least significant bit controls the visibility of the
            first edge, the second bit controls the visibility of the second edge, and the third
            bit controls the visibility of the third edge. This limits the maximum number of
            vertices to 268435455.

    Returns:
        The rasterized image of triangle indices which is represented with an index tensor of a
        shape [N, H, W] of type int32 that stores a triangle ID for each pixel. If a triangle
        covers a pixel and is the closest triangle to the camera, then the pixel will have the
        ID of that triangle. If no triangles cover a pixel, then its ID is -1.

    Note:
        This function is not differentiable. The gradients should be computed with
        :func:`edge_grad_estimator` instead.
    """
    _, index_img = th.ops.rasterize_ext.rasterize(v, vi, height, width, wireframe)
    return index_img


@th.compiler.disable
def rasterize_with_depth(
    v: th.Tensor,
    vi: th.Tensor,
    height: int,
    width: int,
    wireframe: bool = False,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Same as :func:`rasterize` function, additionally returns depth image. Internally it uses the
    same implementation as the rasterize function which still computes depth but does not return
    depth.

    Note:
        The function is not differentiable, including the depth output.

    The split is done intentionally to hide the depth image from the user as it is not
    differentiable which may cause errors if assumed otherwise. Instead, the`barycentrics` function
    should be used instead to
    compute the differentiable version of depth.

    However, we still provide `rasterize_with_depth` which returns non-differentiable depth which
    could allow to avoid call to `barycentrics` function when differentiability is not required.

    Returns:
        The rasterized image of triangle indices of shape [N, H, W] and a depth image of shape
        [N, H, W]. Values in of pixels in the depth image not covered by any pixel are 0.

    """
    depth_img, index_img = th.ops.rasterize_ext.rasterize(
        v, vi, height, width, wireframe
    )
    return depth_img, index_img

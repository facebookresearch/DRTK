# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from . import utils  # noqa  # noqa
from .edge_grad_estimator import edge_grad_estimator, edge_grad_estimator_ref  # noqa
from .filter2d import (  # noqa  # noqa
    downsample,
    filter,
    FilterOptions,
    FilterType,
    low_pass_filter,
    make_resampling_kernel,
    resample_filter,
    upsample,
)
from .grid_scatter import grid_scatter, grid_scatter_ref  # noqa
from .interpolate import (  # noqa
    interpolate,
    interpolate_ref,
    interpolation_matrix,
    interpolation_normal_matrix,
)
from .mipmap_grid_sample import mipmap_grid_sample, mipmap_grid_sample_ref  # noqa
from .msi import msi  # noqa
from .pixel_grid_mesh import (  # noqa
    bernoulli_render_loss,
    bernoulli_topology_samples,
    pixel_grid_mesh,
    pixel_grid_mesh_ref,
)
from .rasterize import rasterize, rasterize_with_depth  # noqa
from .render import render, render_ref  # noqa
from .transform import transform, transform_with_v_cam  # noqa
from .transparent_render import (  # noqa
    render_transparent,
    render_transparent_mlp,
    render_transparent_mlp_ref,
    render_transparent_mlp_two_textures,
    render_transparent_mlp_two_textures_ref,
    render_transparent_ref,
)

__version__ = "0.1.0"  # noqa

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch as th

from .. import rasterizer_ext

try:
    th.classes.load_library(rasterizer_ext.__file__)
    # pyre-fixme[5]: Global expression must be annotated.
    VulkanRasterizerContext = th.classes.rasterizer_ext.VulkanRasterizerContext
    has_vulkan = True
except Exception:
    has_vulkan = False

from .rasterize_function import rasterize, rasterize_packed  # noqa

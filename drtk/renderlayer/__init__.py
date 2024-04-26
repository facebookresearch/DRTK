# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


class settings:
    use_python_renderer = False
    use_vulkan = False
    use_precise_uv_grads = False


from .msi import msi  # noqa
from .projection import DISTORTION_MODES, project_points  # noqa
from .renderlayer import RenderLayer  # noqa

# Will fail if rpack isn't installed.
try:
    from .multimesh_renderlayer import (  # noqa  # noqa
        make_multimesh_from_objs,
        MultiMeshRenderLayer,
    )
except ImportError:
    pass

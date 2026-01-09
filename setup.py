# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
import re
import sys

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def main(debug: bool) -> None:
    extra_link_args = {
        "linux": ["-static-libgcc"] + ([] if debug else ["-flto"]),
        "win32": ["/DEBUG"] if debug else [],
    }
    cxx_args = {
        "linux": ["-std=c++17", "-Wall"]
        + (["-O0", "-g3", "-DDEBUG"] if debug else ["-O3", "--fast-math"]),
        "win32": (
            ["/MT", "/GR-", "/EHsc", '/D "NOMINMAX"']
            + (
                ["/Od", '/D "_DEBUG"']
                if debug
                else ["/O2", "/fp:fast", "/GL", '/D "NDEBUG"']
            )
        ),
    }

    nvcc_args = ["-O0", "-g", "-DDEBUG"] if debug else ["-O3", "--use_fast_math"]
    if not os.getenv("TORCH_CUDA_ARCH_LIST"):
        # Respect TORCH_CUDA_ARCH_LIST when set, otherwise fall back to a default list of archs
        nvcc_args.extend(
            [
                "-gencode=arch=compute_72,code=sm_72",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_90,code=sm_90",
            ]
        )

    # There is som issue effecting latest NVCC and pytorch 2.3.0 https://github.com/pytorch/pytorch/issues/122169
    # The workaround is adding -std=c++20 to NVCC args
    nvcc_args.append("-std=c++17")

    def get_dist(name):
        try:
            return get_distribution(name)
        except DistributionNotFound:
            return None

    root_path = os.path.dirname(os.path.abspath(__file__))
    package_name = "drtk"
    with open(os.path.join(root_path, "drtk", "__init__.py")) as f:
        init_file = f.read()

    pattern = re.compile(r"__version__\s*=\s*\"(\d*\.\d*.\d*)\"")
    groups = pattern.findall(init_file)
    assert len(groups) == 1
    version = groups[0]

    target_os = "none"

    if sys.platform == "darwin":
        target_os = "macos"
    elif os.name == "posix":
        target_os = "linux"
    elif platform.system() == "Windows":
        target_os = "win32"

    if target_os == "none":
        raise RuntimeError("Could not detect platform")
    if target_os == "macos":
        raise RuntimeError("Platform is not supported")

    include_dir = [os.path.join(root_path, "src", "include")]

    with open("README.md") as f:
        readme = f.read()

    pillow = "pillow" if get_dist("pillow-simd") is None else "pillow-simd"

    setup(
        name=package_name,
        version=version,
        author="Reality Labs, Meta",
        description="Differentiable Rendering Toolkit",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="MIT",
        install_requires=["numpy", "torch", "torchvision", pillow],
        ext_modules=[
            CUDAExtension(
                name="drtk.rasterize_ext",
                sources=[
                    "src/rasterize/rasterize_module.cpp",
                    "src/rasterize/rasterize_kernel.cu",
                ],
                extra_compile_args={"cxx": cxx_args[target_os], "nvcc": nvcc_args},
                extra_link_args=extra_link_args[target_os],
                include_dirs=include_dir,
            ),
            CUDAExtension(
                "drtk.render_ext",
                sources=["src/render/render_kernel.cu", "src/render/render_module.cpp"],
                extra_compile_args={"cxx": cxx_args[target_os], "nvcc": nvcc_args},
                include_dirs=include_dir,
            ),
            CUDAExtension(
                "drtk.edge_grad_ext",
                sources=[
                    "src/edge_grad/edge_grad_module.cpp",
                    "src/edge_grad/edge_grad_kernel.cu",
                ],
                extra_compile_args={"cxx": cxx_args[target_os], "nvcc": nvcc_args},
                include_dirs=include_dir,
            ),
            CUDAExtension(
                "drtk.mipmap_grid_sampler_ext",
                sources=[
                    "src/mipmap_grid_sampler/mipmap_grid_sampler_module.cpp",
                    "src/mipmap_grid_sampler/mipmap_grid_sampler_kernel.cu",
                ],
                extra_compile_args={"cxx": cxx_args[target_os], "nvcc": nvcc_args},
                include_dirs=include_dir,
            ),
            CUDAExtension(
                "drtk.msi_ext",
                sources=[
                    "src/msi/msi_module.cpp",
                    "src/msi/msi_kernel.cu",
                ],
                extra_compile_args={"cxx": cxx_args[target_os], "nvcc": nvcc_args},
                include_dirs=include_dir,
            ),
            CUDAExtension(
                "drtk.interpolate_ext",
                sources=[
                    "src/interpolate/interpolate_module.cpp",
                    "src/interpolate/interpolate_kernel.cu",
                ],
                extra_compile_args={"cxx": cxx_args[target_os], "nvcc": nvcc_args},
                include_dirs=include_dir,
            ),
            CUDAExtension(
                "drtk.grid_scatter_ext",
                sources=[
                    "src/grid_scatter/grid_scatter_module.cpp",
                    "src/grid_scatter/grid_scatter_kernel.cu",
                ],
                extra_compile_args={"cxx": cxx_args[target_os], "nvcc": nvcc_args},
                include_dirs=include_dir,
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        packages=["drtk", "drtk.utils"],
    )


if __name__ == "__main__":
    main(any(x in sys.argv for x in ["debug", "-debug", "--debug"]))

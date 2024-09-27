# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import importlib

import torch as th


def load_torch_ops(extension: str) -> None:
    try:
        module = importlib.import_module(extension)
        th.ops.load_library(module.__file__)
    except ImportError as e:
        import sys

        # If running in sphinx, don't raise an error. That way we can build documentation without
        # building extensions
        if "sphinx" in sys.modules:
            return

        raise e

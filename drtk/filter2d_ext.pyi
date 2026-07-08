# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import device as Device, Tensor

def resample_filter(
    x: Tensor, f: Tensor, up: int, down: int, reflect: bool
) -> Tensor: ...
def low_pass_filter(
    x: Tensor,
    n: int,
    freq_div: float,
    alias_guard_band: float,
    filter_type: int,
    reflect: bool,
) -> Tensor: ...
def downsample(
    x: Tensor, n: int, m: int, alias_guard_band: float, filter_type: int, reflect: bool
) -> Tensor: ...
def upsample(
    x: Tensor, n: int, m: int, alias_guard_band: float, filter_type: int, reflect: bool
) -> Tensor: ...
def make_resampling_kernel(
    n: int,
    m: int,
    freq_div: float,
    gain: float,
    alias_guard_band: float,
    filter_type: int,
    d: Device,
) -> Tensor: ...

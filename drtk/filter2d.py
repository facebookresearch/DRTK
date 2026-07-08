# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from enum import Enum
from typing import Callable, cast, Optional, TypeVar

import torch as th
from drtk.utils import load_torch_ops


__all__ = [
    "FilterType",
    "FilterOptions",
    "resample_filter",
    "filter",
    "low_pass_filter",
    "downsample",
    "upsample",
    "make_resampling_kernel",
]


load_torch_ops("drtk.filter2d_ext")

# pyre-fixme[9]: _resample_filter has type `(Tensor, Tensor, int, int, bool) ->
#  Tensor`; used as `OpOverloadPacket`.
_resample_filter: Callable[[th.Tensor, th.Tensor, int, int, bool], th.Tensor] = (
    th.ops.filter2d_ext.resample_filter
)
# pyre-fixme[9]: _upsample has type `(Tensor, int, int, float, int, bool) ->
#  Tensor`; used as `OpOverloadPacket`.
_upsample: Callable[[th.Tensor, int, int, float, int, bool], th.Tensor] = (
    th.ops.filter2d_ext.upsample
)
# pyre-fixme[9]: _downsample has type `(Tensor, int, int, float, int, bool) ->
#  Tensor`; used as `OpOverloadPacket`.
_downsample: Callable[[th.Tensor, int, int, float, int, bool], th.Tensor] = (
    th.ops.filter2d_ext.downsample
)
# pyre-fixme[9]: _low_pass_filter has type `(Tensor, int, float, float, int, bool) ->
#  Tensor`; used as `OpOverloadPacket`.
_low_pass_filter: Callable[[th.Tensor, int, float, float, int, bool], th.Tensor] = (
    th.ops.filter2d_ext.low_pass_filter
)
# pyre-fixme[9]: _make_resampling_kernel has type `(int, int, float, float, float,
#  int, device) -> Tensor`; used as `OpOverloadPacket`.
_make_resampling_kernel: Callable[
    [int, int, float, float, float, int, th.device], th.Tensor
] = th.ops.filter2d_ext.make_resampling_kernel

_FunctionT = TypeVar("_FunctionT", bound=Callable[..., object])


def _compiler_disable(fn: _FunctionT) -> _FunctionT:
    return cast(_FunctionT, th.compiler.disable(fn))


def _use_reflection_padding(padding_mode: str) -> bool:
    if padding_mode == "reflection":
        return True
    if padding_mode == "zeros":
        return False
    raise NotImplementedError(
        "filter2d: expected padding_mode to be 'zeros' or 'reflection', "
        f"but got: {padding_mode!r}"
    )


class FilterType(Enum):
    """Filter families supported by :func:`make_resampling_kernel`."""

    Kaiser = 0
    Lanczos = 1


class FilterOptions:
    """Options used to construct filter2d resampling kernels."""

    __slots__ = ("n_taps", "filter_type", "alias_suppression_level")

    def __init__(
        self,
        n_taps: int = 6,
        filter_type: FilterType = FilterType.Kaiser,
        alias_suppression_level: float = 0.0,
    ) -> None:
        """
        Args:
            n_taps: Number of taps. Default is ``6``. This is not the size of
                the filter tensor: each output pixel is affected by
                ``n_taps`` input pixels during upsampling, and each input pixel
                affects ``n_taps`` output pixels during downsampling. The
                filter tensor size is ``m * n_taps``.
            filter_type: Filter family to build. Default is
                :attr:`FilterType.Kaiser`.
            alias_suppression_level: Value in ``[0, 1]``. Default is ``0.0``.
                ``0`` uses critical sampling, with the cutoff frequency set to
                the bandlimit. ``1`` shifts all alias frequencies above the
                bandlimit into the stopband.
        """
        self.n_taps = n_taps
        self.filter_type = filter_type
        self.alias_suppression_level = alias_suppression_level


@_compiler_disable
def resample_filter(
    x: th.Tensor,
    f: th.Tensor,
    up: int = 1,
    down: int = 1,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """Resample an NCHW tensor with a separable 1D filter.

    The input is upsampled by interleaving zeros, convolved with ``f`` along
    both spatial dimensions, and downsampled by dropping sample points. The
    operation is fused into one CUDA kernel.

    Args:
        x: Input tensor with shape ``(N, C, H, W)``.
        f: 1D filter tensor.
        up: Upsampling factor. Default is ``1``, which leaves the input
            sampling rate unchanged.
        down: Downsampling factor. Default is ``1``, which leaves the output
            sampling rate unchanged.
        padding_mode: Border handling. Supported values are ``"zeros"`` and
            ``"reflection"``. Default is ``"reflection"``.
    """

    return _resample_filter(
        x.contiguous(),
        f.contiguous(),
        up,
        down,
        _use_reflection_padding(padding_mode),
    )


@_compiler_disable
def filter(
    x: th.Tensor,
    f: th.Tensor,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """Filter an NCHW tensor without changing its spatial size.

    Args:
        x: Input tensor with shape ``(N, C, H, W)``.
        f: 1D filter tensor.
        padding_mode: Border handling. Supported values are ``"zeros"`` and
            ``"reflection"``. Default is ``"reflection"``.
    """

    return _resample_filter(
        x.contiguous(),
        f.contiguous(),
        1,
        1,
        _use_reflection_padding(padding_mode),
    )


@_compiler_disable
def upsample(
    x: th.Tensor,
    filter_options: FilterOptions,
    upsample_factor: int = 2,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """Upsample an NCHW tensor by ``upsample_factor``.

    This is the fused equivalent of :func:`make_resampling_kernel` followed by
    :func:`resample_filter`.

    Args:
        x: Input tensor with shape ``(N, C, H, W)``.
        filter_options: Interpolation filter options.
        upsample_factor: Upsampling factor. Must be at least ``1``. Default is
            ``2``.
        padding_mode: Border handling. Supported values are ``"zeros"`` and
            ``"reflection"``. Default is ``"reflection"``.
    """

    return _upsample(
        x.contiguous(),
        filter_options.n_taps,
        upsample_factor,
        filter_options.alias_suppression_level,
        filter_options.filter_type.value,
        _use_reflection_padding(padding_mode),
    )


@_compiler_disable
def downsample(
    x: th.Tensor,
    filter_options: FilterOptions,
    downsample_factor: int = 2,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """Downsample an NCHW tensor by ``downsample_factor``.

    This is the fused equivalent of :func:`make_resampling_kernel` followed by
    :func:`resample_filter`.

    Args:
        x: Input tensor with shape ``(N, C, H, W)``.
        filter_options: Interpolation filter options.
        downsample_factor: Downsampling factor. Must be at least ``1``. Default
            is ``2``.
        padding_mode: Border handling. Supported values are ``"zeros"`` and
            ``"reflection"``. Default is ``"reflection"``.
    """

    return _downsample(
        x.contiguous(),
        filter_options.n_taps,
        downsample_factor,
        filter_options.alias_suppression_level,
        filter_options.filter_type.value,
        _use_reflection_padding(padding_mode),
    )


@_compiler_disable
def low_pass_filter(
    x: th.Tensor,
    filter_options: FilterOptions,
    freq_div: float = 1.0,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """Low-pass filter an NCHW tensor without changing its spatial size.

    Args:
        x: Input tensor with shape ``(N, C, H, W)``.
        filter_options: Interpolation filter options.
        freq_div: Frequency divider. The cutoff frequency is reduced by this
            factor. Default is ``1.0``.
        padding_mode: Border handling. Supported values are ``"zeros"`` and
            ``"reflection"``. Default is ``"reflection"``.
    """

    return _low_pass_filter(
        x.contiguous(),
        filter_options.n_taps,
        freq_div,
        filter_options.alias_suppression_level,
        filter_options.filter_type.value,
        _use_reflection_padding(padding_mode),
    )


@_compiler_disable
def make_resampling_kernel(
    filter_options: FilterOptions,
    m: int = 1,
    freq_div: float = 1.0,
    gain: float = 1.0,
    device: Optional[th.device] = None,
) -> th.Tensor:
    """Build a 1D low-pass resampling filter.

    Args:
        filter_options: Interpolation filter options.
        m: Upsampling or downsampling factor. Default is ``1``.
        freq_div: Frequency divider. The cutoff frequency is reduced by this
            factor. Default is ``1.0``.
        gain: Kernel values sum to this value. Use ``gain == m`` when
            upsampling to preserve signal magnitude. Default is ``1.0``.
        device: Device for the returned filter tensor. Default is CPU.
    """

    if device is None:
        device = th.device("cpu")
    if not isinstance(device, th.device):
        device = th.device(device)

    return _make_resampling_kernel(
        filter_options.n_taps,
        m,
        freq_div,
        gain,
        filter_options.alias_suppression_level,
        filter_options.filter_type.value,
        device,
    )

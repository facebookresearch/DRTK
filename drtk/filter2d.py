# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import builtins
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


def _is_sphinx_build() -> bool:
    return bool(getattr(builtins, "__sphinx_build__", False))


def _missing_filter2d_ext(*args: object, **kwargs: object) -> th.Tensor:
    raise ImportError(
        "drtk.filter2d_ext is required to call filter2d functions. "
        "Build and install the DRTK extensions before using this API."
    )


def _filter2d_op(name: str) -> Callable[..., th.Tensor]:
    if _is_sphinx_build() and not hasattr(th.ops.filter2d_ext, name):
        return _missing_filter2d_ext
    return getattr(th.ops.filter2d_ext, name)


_resample_filter = cast(
    Callable[[th.Tensor, th.Tensor, int, int, bool], th.Tensor],
    _filter2d_op("resample_filter"),
)
_upsample = cast(
    Callable[[th.Tensor, int, int, float, int, bool], th.Tensor],
    _filter2d_op("upsample"),
)
_downsample = cast(
    Callable[[th.Tensor, int, int, float, int, bool], th.Tensor],
    _filter2d_op("downsample"),
)
_low_pass_filter = cast(
    Callable[[th.Tensor, int, float, float, int, bool], th.Tensor],
    _filter2d_op("low_pass_filter"),
)
_make_resampling_kernel = cast(
    Callable[[int, int, float, float, float, int, th.device], th.Tensor],
    _filter2d_op("make_resampling_kernel"),
)

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


def _validate_filter_type(filter_type: object) -> FilterType:
    if not isinstance(filter_type, FilterType):
        raise TypeError(
            f"filter2d: filter_type must be a FilterType value, but got {filter_type!r}"
        )
    return filter_type


class FilterOptions:
    """Options used to construct filter2d resampling kernels."""

    __slots__ = ("n_taps", "filter_type", "alias_guard_band")

    def __init__(
        self,
        n_taps: int = 6,
        filter_type: FilterType = FilterType.Kaiser,
        alias_guard_band: Optional[float] = None,
        alias_suppression_level: Optional[float] = None,
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
            alias_guard_band: Cutoff placement knob for the alias-free GAN
                low-pass filter design. Must be non-negative; the recommended
                range is ``[0, 1]``. Default is ``0.0`` for compatibility.
                Frequencies are
                normalized to the input sampling rate. For a given
                ``freq_div``, the usable bandlimit is
                ``bandlimit = 0.5 / freq_div`` and the transition half-width is
                ``transition_half_width = (sqrt(2) - 1) * bandlimit``. The
                filter cutoff is placed at
                ``bandlimit - alias_guard_band * transition_half_width``.
                Therefore ``0.0`` puts the cutoff at the bandlimit; this is the
                least blurry setting, but the transition band extends past the
                bandlimit. ``1.0`` puts the cutoff one transition half-width
                below the bandlimit, so the transition upper edge reaches the
                bandlimit. Values between ``0.0`` and ``1.0`` interpolate
                between those placements. Values above ``1.0`` leave extra
                guard band and blur more. This parameter does not directly set
                stopband attenuation; attenuation also depends on ``n_taps``
                and ``filter_type``.
            alias_suppression_level: Backward-compatible alias for
                ``alias_guard_band``.
        """
        alias_guard_band_value: float
        if alias_guard_band is None:
            alias_guard_band_value = (
                0.0 if alias_suppression_level is None else alias_suppression_level
            )
        else:
            if (
                alias_suppression_level is not None
                and alias_guard_band != alias_suppression_level
            ):
                raise ValueError(
                    "FilterOptions: specify only one of alias_guard_band and "
                    "alias_suppression_level"
                )
            alias_guard_band_value = alias_guard_band

        self.n_taps = n_taps
        self.filter_type = _validate_filter_type(filter_type)
        self.alias_guard_band = alias_guard_band_value

    @property
    def alias_suppression_level(self) -> float:
        return self.alias_guard_band

    @alias_suppression_level.setter
    def alias_suppression_level(self, value: float) -> None:
        self.alias_guard_band = value


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
    CUDA tensors use the fused kernel. CPU tensors use the native ATen fallback
    with the same padding and adjoint alignment.

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
        filter_options.alias_guard_band,
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
        filter_options.alias_guard_band,
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
        filter_options.alias_guard_band,
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
        filter_options.alias_guard_band,
        filter_options.filter_type.value,
        device,
    )

# pyre-strict
from enum import Enum
from typing import Callable, Optional

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
# pyre-fixme[9]: _low_pass_filter has type `(Tensor, int, int, float, int, bool) ->
#  Tensor`; used as `OpOverloadPacket`.
_low_pass_filter: Callable[[th.Tensor, int, int, float, int, bool], th.Tensor] = (
    th.ops.filter2d_ext.low_pass_filter
)
# pyre-fixme[9]: _make_resampling_kernel has type `(int, int, float, float, float,
#  int, device) -> Tensor`; used as `OpOverloadPacket`.
_make_resampling_kernel: Callable[
    [int, int, float, float, float, int, th.device], th.Tensor
] = th.ops.filter2d_ext.make_resampling_kernel


class FilterType(Enum):
    """
    Enum for supported filter types.
    For reference, please see `care.utils.filter2d.filter2d_ref.make_resampling_kernel`
    """

    Kaiser = 0
    Lanczos = 1


class FilterOptions:
    """
    Options holder for filter construction
    """

    __slots__ = ("n_taps", "filter_type", "alias_suppression_level")

    def __init__(
        self,
        n_taps: int = 6,
        filter_type: FilterType = FilterType.Kaiser,
        alias_suppression_level: float = 0.0,
    ) -> None:
        """
        Args:
            n_taps (int): Number of taps. This is not the size of the filter tensor.
                This means that each output pixel is affected by `n` input pixels in upsampling
                and each input pixel affects `n` output pixels in downsampling. The filter tensor size will
                be `m * n`.
            filter_type (FilterType): The type of filter to build, Kaiser or Lanczos.
            alias_suppression_level (float): Requires values in range [0..1]:
                - when 0, critical sampling will be used, e.i. cut-off frequency is set to bandlmit.
                    This is optimal setting for most use-cases.
                - when 1, cut-off frequency is set s.t. all alias frequencies (above bandlmit) are in the
                    stopband. This setting is useful when it is needed to completely eliminate aliasing.
                Default is 0.
        """
        self.n_taps = n_taps
        self.filter_type = filter_type
        self.alias_suppression_level = alias_suppression_level


def resample_filter(
    x: th.Tensor,
    f: th.Tensor,
    up: int = 1,
    down: int = 1,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """
    Input tensor will be upsampled `up` times by interleaving with zeros, convolved with `f` along `H`
    and `W` dimension and downsampled `down` times by dropping sample points.
    Everythng is fused into operation running in one CUDA kernel.
    For reference, please see `care.utils.filter2d.filter2d_ref.resample_filter`

    Args:
        x (Tensor): Input tensor of shape NCHW.
        f (Tensor): filter to convolve with. Must be 1D tensor.
        up (int): Upsampling factor. Default 1 - no unsampling
        down (int): Downsampling factor. Default 1 - no downsampling
        padding_mode (str):  'zeros', 'border', or 'reflection'. Determines how the border is treated. The
            best results are achieved with  'reflection'. Default 'reflection'
    """

    if padding_mode == "reflection":
        reflection_pad = True
    elif padding_mode == "zeros":
        reflection_pad = False
    else:
        raise NotImplementedError(f"Unknown padding_mode: {padding_mode}")

    return _resample_filter(x.contiguous(), f.contiguous(), up, down, reflection_pad)


def filter(
    x: th.Tensor,
    f: th.Tensor,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """
    Same as `resample_filter`, but does not change the size of the input.

    Args:
        x (Tensor): Input tensor of shape NCHW.
        f (Tensor): filter to convolve with. Must be 1D tensor.
        padding_mode (str):  'zeros', 'border', or 'reflection'. Determines how the border is treated. The
            best results are achieved with  'reflection'. Default 'reflection'
    """

    if padding_mode == "reflection":
        reflection_pad = True
    elif padding_mode == "zeros":
        reflection_pad = False
    else:
        raise NotImplementedError(f"Unknown padding_mode: {padding_mode}")

    return _resample_filter(x.contiguous(), f.contiguous(), 1, 1, reflection_pad)


def upsample(
    x: th.Tensor,
    filter_options: FilterOptions,
    upsample_factor: int = 2,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """
    Upsamples the input by factor of `upsample_factor`.
    Alias for `make_resampling_kernel` + `resample_filter`.

    Args:
        x (th.Tensor): Input tensor.
        filter_options (FilterOptions): Interpolation filter options.
        upsample_factor (int): upsample factor to be used. Must be >= 1. Default is 2.
        padding_mode (str): see `resample_filter`. Default is "reflection".
    """

    if padding_mode == "reflection":
        reflection_pad = True
    elif padding_mode == "zeros":
        reflection_pad = False
    else:
        raise NotImplementedError(f"Unknown padding_mode: {padding_mode}")

    return _upsample(
        x.contiguous(),
        filter_options.n_taps,
        upsample_factor,
        filter_options.alias_suppression_level,
        filter_options.filter_type.value,
        reflection_pad,
    )


def downsample(
    x: th.Tensor,
    filter_options: FilterOptions,
    downsample_factor: int = 2,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """
    Downsamples the input by factor of `downsample_factor`.
    Alias for `make_resampling_kernel` + `resample_filter`.

    Args:
        x (th.Tensor): Input tensor.
        filter_options (FilterOptions): Interpolation filter options.
        downsample_factor (int): downsample factor to be used. Must be >= 1. Default is 2.
        padding_mode (str): see `resample_filter`. Default is "reflection".
    """

    if padding_mode == "reflection":
        reflection_pad = True
    elif padding_mode == "zeros":
        reflection_pad = False
    else:
        raise NotImplementedError(f"Unknown padding_mode: {padding_mode}")

    return _downsample(
        x.contiguous(),
        filter_options.n_taps,
        downsample_factor,
        filter_options.alias_suppression_level,
        filter_options.filter_type.value,
        reflection_pad,
    )


def low_pass_filter(
    x: th.Tensor,
    filter_options: FilterOptions,
    freq_div: int = 1,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """
    Does not change the size of the input.
    Alias for `make_resampling_kernel` + `resample_filter`.

    Args:
        x (th.Tensor): Input tensor.
        filter_options (FilterOptions): Interpolation filter options.
        freq_div (float): Frequency divider. Cut-off frequency will be reduced by this factor. Default is 1.
        padding_mode (str): see `resample_filter`. Default is "reflection".
    """

    if padding_mode == "reflection":
        reflection_pad = True
    elif padding_mode == "zeros":
        reflection_pad = False
    else:
        raise NotImplementedError(f"Unknown padding_mode: {padding_mode}")

    return _low_pass_filter(
        x.contiguous(),
        filter_options.n_taps,
        freq_div,
        filter_options.alias_suppression_level,
        filter_options.filter_type.value,
        reflection_pad,
    )


def make_resampling_kernel(
    filter_options: FilterOptions,
    m: int = 1,
    freq_div: float = 1.0,
    gain: float = 1.0,
    device: Optional[th.device] = None,
) -> th.Tensor:
    """
    Returns low-pass filter

    Args:
        filter_options (FilterOptions): Interpolation filter options.
        m (int): Upsampling (downsampling) factor.
        freq_div (float): Frequency divider. Cut-off frequency will be reduced by this factor. Default is 1.
        gain (float): Kernel values will add-up to this value. It is used to keep the signal magnitude
            unchanged by setting `gain == m` when upsampling.
        device (th.device): the device to be used for the kernel tensor.
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

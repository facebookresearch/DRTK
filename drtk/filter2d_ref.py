# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch as th
import torch.nn.functional as thf
from drtk.filter2d import FilterOptions, FilterType
from torchvision.transforms import functional as F

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


def check_padding_mode(padding_mode: str) -> None:
    if (
        padding_mode != "zeros"
        and padding_mode != "border"
        and padding_mode != "reflection"
    ):
        raise ValueError(
            "filter2d.resample_filter(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            "but got: '{}'".format(padding_mode)
        )


def resample_filter(
    x: th.Tensor,
    f: th.Tensor,
    up: int = 1,
    down: int = 1,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """Pure PyTorch reference implementation of :func:`drtk.filter2d.resample_filter`.

    The input is upsampled by interleaving zeros, convolved with ``f`` along
    both spatial dimensions, and downsampled by dropping sample points.

    Args:
        x: Input tensor with shape ``(N, C, H, W)``.
        f: 1D filter tensor.
        up: Upsampling factor. Default is ``1``, which leaves the input
            sampling rate unchanged.
        down: Downsampling factor. Default is ``1``, which leaves the output
            sampling rate unchanged.
        padding_mode: Border handling. The reference supports ``"zeros"``,
            ``"border"``, and ``"reflection"``; the fused native op supports
            ``"zeros"`` and ``"reflection"``. Default is ``"reflection"``.
    """
    assert isinstance(x, th.Tensor) and x.ndim == 4
    assert isinstance(f, th.Tensor) and f.ndim == 1

    check_padding_mode(padding_mode)

    # `thf.pad` uses slightly different mode names than `grid_sample` and
    # related functions. Keep the grid_sample names in the public API and
    # translate them here.
    padding_mode_alt = {
        "zeros": "constant",
        "border": "replicate",
        "reflection": "reflect",
    }[padding_mode]

    n = f.shape[0]
    num_channels = x.shape[1]

    padx_0, padx_1, pady_0, pady_1 = (
        calc_pad_0(up, down, n),
        calc_pad_1(up, down, n),
        calc_pad_0(up, down, n),
        calc_pad_1(up, down, n),
    )

    if padding_mode == "zeros":
        # if padding_mode is 'zeros' then the implementation is simpler: insert zeros -> pad -> convolve
        x = insert_zeros(x, up)
        x = thf.pad(x, [padx_0, padx_1, pady_0, pady_1], mode=padding_mode_alt)
    else:
        # For 'border' or 'reflection', pad before upsampling. This causes
        # over-padding, so crop after upsampling: pad -> insert zeros -> crop
        # -> convolve.

        x = thf.pad(
            x,
            [
                ceildiv(padx_0, up),
                ceildiv(padx_1, up),
                ceildiv(pady_0, up),
                ceildiv(pady_1, up),
            ],
            mode=padding_mode_alt,
        )

        x = insert_zeros(x, up)

        # crop if needed
        if any(x % up != 0 for x in [padx_0, padx_1, pady_0, pady_1]):
            m = [ceildiv(x, up) * up - x for x in [padx_0, padx_1, pady_0, pady_1]]
            x = x[:, :, m[2] : x.shape[-2] - m[3], m[0] : x.shape[-1] - m[1]]

    # Convolve with the filter and drop sample points using stride.
    f = f[None, None, ...].repeat([num_channels, 1, 1])
    x = thf.conv2d(
        input=x, weight=f.unsqueeze(2), groups=num_channels, stride=(1, down)
    )
    x = thf.conv2d(
        input=x, weight=f.unsqueeze(3), groups=num_channels, stride=(down, 1)
    )
    return x


def filter(
    x: th.Tensor,
    f: th.Tensor,
    padding_mode: str = "reflection",
) -> th.Tensor:
    """Pure PyTorch reference implementation of :func:`drtk.filter2d.filter`.

    Args:
        x: Input tensor with shape ``(N, C, H, W)``.
        f: 1D filter tensor.
        padding_mode: Border handling. See :func:`resample_filter`. Default is
            ``"reflection"``.
    """

    return resample_filter(x, f, 1, 1, padding_mode)


def ceildiv(a: int, b: int) -> int:
    """Return ``ceil(a / b)`` for integers."""
    return -(a // -b)


def insert_zeros(x: th.Tensor, up: int) -> th.Tensor:
    assert x.ndim == 4
    assert up >= 1
    if up == 1:
        return x
    batch_size, num_channels, in_height, in_width = x.shape
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = thf.pad(x, [0, up - 1, 0, 0, 0, up - 1])
    x = x.reshape([batch_size, num_channels, in_height * up, in_width * up])
    return x


def calc_pad_0(up: int, down: int, n: int) -> int:
    """Compute left padding"""
    if down == 1 and up == 1:
        return n // 2
    else:
        if down != 1:
            return (n - down + 1) // 2
        else:
            return (n + up - 1) // 2


def calc_pad_1(up: int, down: int, n: int) -> int:
    """Compute right padding"""
    if down == 1 and up == 1:
        return (n - 1) // 2
    else:
        if down != 1:
            return (n - down) // 2
        else:
            return (n - up) // 2


def make_kernel_kaiser(
    n: int, fh_s: float, fc_s: float, m: int, gain: float
) -> npt.NDArray:
    """
    Returns Kaiser low-pass filter

    Args:
        n (int): Number of taps. This is not the size of the tensor that is returned.
            This means that each output pixel is affected by `n` input pixels in upsampling
            and each input pixel affects `n` output pixels in downsampling. The returned tensor size will
            be `m * n`.
        fh_s (float): Transition band half-width as fraction of sampling rate.
        fc_s (float): Cut-off frequency as fraction of sampling rate.
        m (int): Upsampling (downsampling) factor.
        gain (float): Kernel values will add-up to this value. It is used to keep the signal magnitude
            unchanged by setting `gain == m` when upsampling.
    """

    # Notation:
    # fh - transition band half-width
    # fc - cut-off frequency
    # s - input sampling rate.
    # s' - output sampling rate. s' = s * m

    # kernel size: n' = n * m
    n_p = n * m

    # Spatial extent: L' = (n' - 1) / m
    L_p = (n_p - 1) / m

    # Compute the width of the transition band as a fraction of s'/2.
    # \Delta f' = (2 f_h ) / (s' / 2)
    df_p = (2 * fh_s) / (m / 2)

    # Kaiser formulas
    A = 2.285 * (n_p - 1) * np.pi * df_p + 7.95
    beta = (
        (0.1102 * (A - 8.7))
        if A > 50
        else (0 if A < 21 else (0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)))
    )

    # coordinates of taps
    x = np.linspace(0, n_p - 1, n_p)
    x = (x - (n_p - 1) / 2) / m

    # window
    w = np.i0(beta * (1.0 - (2 * x / L_p) ** 2) ** 0.5) / np.i0(beta)

    # kernel = sinc * window
    kern = w * 2 * fc_s * np.sinc(2 * fc_s * x)

    # due to the window, the filter may not add up to 1, so we need to normalize it
    # also multiplying by gain
    kern = kern / kern.sum() * gain
    return kern


def make_kernel_lanczos(n: int, fc_s: float, m: int, gain: float) -> npt.NDArray:
    """
    Returns lanczos low-pass filter. See `make_kernel_kaiser` for more details.

    """
    n_p = n * m
    x = np.linspace(0, n_p - 1, n_p)
    x = (x - (n_p - 1) / 2) / m

    # parameter `a` is picked such that the resulting filter has exactly `n * m` non zero values.
    a = np.ceil(2.0 * fc_s * (float(n_p) - 1.0) / 2.0 / float(m))

    # The part `(np.abs(2 * fc_s * x) < a)` actually is not needed, because we picked `a` such that all
    # locations will be within `a` radius (but it is kept to reflect the original formula).
    kern = (
        np.sinc(2 * fc_s * x) * np.sinc(2 * fc_s * x / a) * (np.abs(2 * fc_s * x) < a)
    )
    kern = kern / kern.sum() * gain
    return kern


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

    # We pick distance between pixels to be 1, so that input sampling rate is
    # also 1, and the bandlimit is 1/2. Following StyleGAN3, the transition
    # half-width is (sqrt(2) - 1) * bandlimit. `alias_guard_band` moves the
    # cutoff inward by that many transition half-widths.
    fh_s = (2**0.5 - 1) / 2 / freq_div

    fc_s = 1 / 2 / freq_div - fh_s * filter_options.alias_guard_band
    if filter_options.filter_type == FilterType.Kaiser:
        f = make_kernel_kaiser(filter_options.n_taps, fh_s, fc_s, m=m, gain=gain)
    elif filter_options.filter_type == FilterType.Lanczos:
        f = make_kernel_lanczos(filter_options.n_taps, fc_s, m=m, gain=gain)
    else:
        raise RuntimeError(f"Unknown filter_type: {filter_options.filter_type}")

    return th.as_tensor(f, dtype=th.float32, device=device)


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

    f = make_resampling_kernel(
        filter_options,
        upsample_factor,
        1.0,
        upsample_factor,
        x.device,
    )
    return resample_filter(x, f, upsample_factor, 1, padding_mode)


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
    f = make_resampling_kernel(
        filter_options,
        downsample_factor,
        1.0,
        1.0,
        x.device,
    )
    return resample_filter(x, f, 1, downsample_factor, padding_mode)


def low_pass_filter(
    x: th.Tensor,
    filter_options: FilterOptions,
    freq_div: float = 1.0,
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
    f = make_resampling_kernel(
        filter_options,
        1,
        freq_div,
        1.0,
        x.device,
    )
    return resample_filter(x, f, 1, 1, padding_mode)


def fast_gaussian_blur(img: th.Tensor, kernel_size: int, sigma: float) -> th.Tensor:
    """
    Applies Gaussian blur to the input image. Similar as
    torchvision.transforms.functional.gaussian_blur, but
    we use kernel separability for faster computation.

    Args:
        img (th.Tensor): Input tensor of shape [..., H, W]
        kernel_size (int): Kernel size. Must be odd and >= 1
        sigma (float): Standard deviation value for gaussian kernel.
    Returns:
        th.Tensor: Blurred output with same size and type as `img`.
    """

    img_x_blur = F.gaussian_blur(
        img=img,
        kernel_size=[1, kernel_size],
        sigma=[
            sigma,
        ],
    )
    return F.gaussian_blur(
        img=img_x_blur,
        kernel_size=[kernel_size, 1],
        sigma=[
            sigma,
        ],
    )

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>

enum class GridSamplerInterpolation { Bilinear, Nearest, Bicubic };

using at::native::clip_coordinates;
using at::native::cubic_interp1d;
using at::native::fastAtomicAdd;
using at::native::get_cubic_upsampling_coefficients;
using at::native::grid_sampler_compute_source_index;
using at::native::grid_sampler_compute_source_index_set_grad;
using at::native::grid_sampler_unnormalize;
using at::native::grid_sampler_unnormalize_set_grad;
using at::native::reflect_coordinates;
using at::native::safe_downgrade_to_int_range;
using at::native::within_bounds_2d;
using at::native::detail::GridSamplerPadding;

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
area_pixel_compute_source_index(scalar_t scale, int64_t dst_index, bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    scalar_t src_idx = scale * (dst_index + 0.5) - 0.5;
    return (src_idx < 0) ? scalar_t(0) : src_idx;
  }
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
area_pixel_compute_scale(int64_t input_size, int64_t output_size, bool align_corners) {
  // see Note [area_pixel_compute_scale]
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<scalar_t>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<scalar_t>(0);
    }
  } else {
    return static_cast<scalar_t>(input_size) / output_size;
  }
}

template <typename scalar_t, typename index_t>
static __forceinline__ __device__ void safe_add_2d(
    scalar_t* data,
    int h,
    int w,
    int sH,
    int sW,
    int H,
    int W,
    scalar_t delta,
    const index_t NC_offset,
    const index_t memory_span) {
  if (within_bounds_2d(h, w, H, W)) {
    fastAtomicAdd(data, NC_offset + h * sH + w * sW, memory_span, delta, true);
  }
}

template <typename scalar_t, typename index_t>
static __forceinline__ __device__ void add_2d(
    scalar_t* data,
    int h,
    int w,
    int sH,
    int sW,
    scalar_t delta,
    const index_t NC_offset,
    const index_t memory_span) {
  fastAtomicAdd(data, NC_offset + h * sH + w * sW, memory_span, delta, true);
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
compute_coordinates(scalar_t coord, int size, GridSamplerPadding padding_mode, bool align_corners) {
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t get_value_bounded(
    scalar_t* data,
    scalar_t x,
    scalar_t y,
    int W,
    int H,
    int sW,
    int sH,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return static_cast<scalar_t>(0);
}

// Calculate the differential of the cubic convolution, i.e. `d coeff / d x`
template <typename scalar_t>
static __forceinline__ __device__ void get_cubic_coefficients_grad(scalar_t coeffs[4], scalar_t t) {
  // Must be the same as forward calculation in
  // aten/src/ATen/native/cuda/UpSample.cuh:get_cubic_upsample_coefficients
  scalar_t A = -0.75;

  scalar_t x;
  x = -1 - t; // 1 < x = |-1 - tx| < 2
  coeffs[0] = (-3 * A * x - 10 * A) * x - 8 * A;
  x = -t; // x = |0 - tx| <= 1
  coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 1 - t; // x = |1 - tx| <= 1
  coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 2 - t; // 1 < x = |2 - tx| < 2
  coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}

template <typename scalar_t, typename index_t>
static __forceinline__ __device__ void add_value_bounded(
    scalar_t* data,
    scalar_t x,
    scalar_t y,
    int W,
    int H,
    int sW,
    int sH,
    scalar_t delta,
    GridSamplerPadding padding_mode,
    bool align_corners,
    const index_t NC_offset,
    const index_t memory_span) {
  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);

  safe_add_2d(data, iy, ix, sH, sW, H, W, delta, NC_offset, memory_span);
}

template <typename scalar_t, int D>
__device__ __forceinline__ static math::TVec<scalar_t, D> cubic_interp1d(
    math::TVec<scalar_t, D> x0,
    math::TVec<scalar_t, D> x1,
    math::TVec<scalar_t, D> x2,
    math::TVec<scalar_t, D> x3,
    scalar_t t) {
  scalar_t coeffs[4];
  get_cubic_upsampling_coefficients<scalar_t>(coeffs, t);
  using namespace math;

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename scalar_t, typename index_t>
inline __device__ typename math::TVec4<scalar_t> load4(scalar_t* ptr, index_t stride) {
  return {ptr[0 * stride], ptr[1 * stride], ptr[2 * stride], ptr[3 * stride]};
}

template <typename scalar_t, typename index_t>
static __forceinline__ __device__ void safe_add_2d4(
    scalar_t* data,
    index_t stride,
    int h,
    int w,
    int sH,
    int sW,
    int H,
    int W,
    math::TVec4<scalar_t> delta,
    const index_t N_offset,
    const index_t memory_span) {
  if (within_bounds_2d(h, w, H, W)) {
    auto ptr = N_offset + h * sH + w * sW;
    fastAtomicAdd(data, ptr + 0 * stride, memory_span, delta.x, true);
    fastAtomicAdd(data, ptr + 1 * stride, memory_span, delta.y, true);
    fastAtomicAdd(data, ptr + 2 * stride, memory_span, delta.z, true);
    fastAtomicAdd(data, ptr + 3 * stride, memory_span, delta.w, true);
  }
}

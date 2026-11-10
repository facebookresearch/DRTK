// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include <kernel_utils.h>

#include "transparent_render_kernel.h"

namespace {

constexpr int kTileSize = 4;
constexpr int kThreadsPerBlock = 256;
constexpr int kTileThreads = kTileSize * kTileSize;
constexpr int kFacesPerBatch = kTileThreads < 32 ? kTileThreads : 32;
constexpr int kMlpWidth = 8;
constexpr int kMaxMlpLayers = 5;
constexpr int kMaxFeatureChannels = 32;
constexpr int kMaxMlpWeightParams =
    kMlpWidth * kMaxFeatureChannels + (kMaxMlpLayers - 2) * kMlpWidth * kMlpWidth + 4 * kMlpWidth;
constexpr int kMaxMlpBiasParams = (kMaxMlpLayers - 1) * kMlpWidth + 4;
constexpr float kAlphaMax = 0.999f;
constexpr float kExpDensityMax = 15.0f;
constexpr int32_t kActivationModeCount = 4;

enum class TextureInterpolation : int32_t { Bilinear = 0, Nearest = 1, Bicubic = 2 };
enum class TextureRgbActivation : int32_t { Relu = 0, Identity = 1, ShiftedSigmoid = 2, Clamp = 3 };
enum class TextureAlphaActivation : int32_t {
  ReluDensity = 0,
  ShiftedSigmoid = 1,
  ExpDensity = 2,
  Clamp = 3
};
enum class MlpRgbActivation : int32_t {
  ShiftedRelu = 0,
  ShiftedIdentity = 1,
  Sigmoid = 2,
  ShiftedClamp = 3
};
enum class MlpAlphaActivation : int32_t {
  ShiftedReluDensity = 0,
  Sigmoid = 1,
  ExpDensity = 2,
  ShiftedClamp = 3
};

using at::native::clip_coordinates;
using at::native::get_cubic_upsampling_coefficients;
using at::native::grid_sampler_compute_source_index;
using at::native::grid_sampler_compute_source_index_set_grad;
using at::native::grid_sampler_unnormalize;
using at::native::grid_sampler_unnormalize_set_grad;
using at::native::reflect_coordinates;
using at::native::safe_downgrade_to_int_range;
using at::native::within_bounds_2d;
using at::native::detail::GridSamplerPadding;

// T = mantissa * 2^(-exponent). Keeping the product normalized makes
// reverse traversal independent of alpha rounding and float32 underflow.
struct ScaledTransmittance {
  float mantissa = 1.0f;
  int32_t exponent = 0;

  __device__ void multiply(float factor) {
    mantissa *= factor;
    if (mantissa < 0x1p-32f) {
      mantissa *= 0x1p32f;
      exponent += 32;
    }
  }

  __device__ float value() const {
    return ldexpf(mantissa, -exponent);
  }

  __device__ float divide(float factor) {
    mantissa /= factor;
    if (mantissa >= 1.0f && exponent > 0) {
      mantissa *= 0x1p-32f;
      exponent -= 32;
    }
    return value();
  }
};

struct DepthAccumulator {
  float weighted = 0.0f;
  float alpha = 0.0f;
  float median = 0.0f;
  int32_t median_face = -1;

  __device__ void
  add(float transmittance, float opacity, float z, int32_t face, TransparentDepthMode mode) {
    const float weight = transmittance * opacity;
    alpha += weight;
    if (mode == TransparentDepthMode::Median) {
      if (median_face < 0 && alpha >= 0.5f) {
        median = z;
        median_face = face;
      }
    } else {
      weighted += weight * z;
    }
  }

  __device__ float value(float transmittance, const TransparentDepthOptions& options) const {
    switch (options.mode) {
      case TransparentDepthMode::Expected:
        return weighted + transmittance * options.background;
      case TransparentDepthMode::Accumulated:
        return weighted;
      case TransparentDepthMode::ForegroundMean:
        return alpha > options.alpha_threshold ? weighted / alpha : options.background;
      case TransparentDepthMode::Median:
        return median_face >= 0 ? median : options.background;
    }
    return 0.0f;
  }
};

struct TileBins {
  torch::Tensor offsets;
  torch::Tensor face_ids;
};

struct TriangleData {
  int32_t vertex_id[3];
  float x[3];
  float y[3];
  float z[3];
  float u[3];
  float v[3];
  float reference_u[3];
  float reference_v[3];
  float opacity[3];
};

struct GeometryParams {
  const float* vertices;
  const int32_t* faces;
  const float* uv;
  const float* reference_uv;
  const float* opacity;
  int32_t vertex_count;
  int32_t face_count;
};

struct TinyMlpParams {
  const float* weights[kMaxMlpLayers];
  const float* biases[kMaxMlpLayers];
  int32_t input_channels;
  float negative_slope;
  int32_t rgb_activation;
  int32_t alpha_activation;
};

struct TinyMlpGradParams {
  float* weights[kMaxMlpLayers];
  float* biases[kMaxMlpLayers];
};

struct FragmentAttributeGrads {
  float bary[3];
  float u[3];
  float v[3];
  float reference_u[3];
  float reference_v[3];
};

__device__ __forceinline__ float clamp_away_from_zero(float value) {
  constexpr float kEpsilon = 1.0e-12f;
  if (value < 0.0f) {
    return fminf(value, -kEpsilon);
  }
  return fmaxf(value, kEpsilon);
}

__device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float activate_texture_rgb(float value, int32_t activation) {
  switch (static_cast<TextureRgbActivation>(activation)) {
    case TextureRgbActivation::Relu:
      return fmaxf(value, 0.0f);
    case TextureRgbActivation::Identity:
      return value;
    case TextureRgbActivation::ShiftedSigmoid:
      return sigmoid(value - 0.5f);
    case TextureRgbActivation::Clamp:
      return fminf(fmaxf(value, 0.0f), 1.0f);
  }
  return 0.0f;
}

__device__ __forceinline__ float activate_texture_rgb_backward(float value, int32_t activation) {
  switch (static_cast<TextureRgbActivation>(activation)) {
    case TextureRgbActivation::Relu:
      return value > 0.0f ? 1.0f : 0.0f;
    case TextureRgbActivation::Identity:
      return 1.0f;
    case TextureRgbActivation::ShiftedSigmoid: {
      const float output = sigmoid(value - 0.5f);
      return output * (1.0f - output);
    }
    case TextureRgbActivation::Clamp:
      return value >= 0.0f && value <= 1.0f ? 1.0f : 0.0f;
  }
  return 0.0f;
}

__device__ __forceinline__ float activate_texture_alpha(float value, int32_t activation) {
  switch (static_cast<TextureAlphaActivation>(activation)) {
    case TextureAlphaActivation::ReluDensity:
      return -expm1f(-fmaxf(value, 0.0f));
    case TextureAlphaActivation::ShiftedSigmoid:
      return sigmoid(value - 0.5f);
    case TextureAlphaActivation::ExpDensity: {
      const float bounded_value = value > kExpDensityMax ? kExpDensityMax : value;
      return -expm1f(-expf(bounded_value));
    }
    case TextureAlphaActivation::Clamp:
      return fminf(fmaxf(value, 0.0f), 1.0f);
  }
  return 0.0f;
}

__device__ __forceinline__ float activate_texture_alpha_backward(float value, int32_t activation) {
  switch (static_cast<TextureAlphaActivation>(activation)) {
    case TextureAlphaActivation::ReluDensity:
      return value > 0.0f ? expf(-value) : 0.0f;
    case TextureAlphaActivation::ShiftedSigmoid: {
      const float output = sigmoid(value - 0.5f);
      return output * (1.0f - output);
    }
    case TextureAlphaActivation::ExpDensity: {
      if (value > kExpDensityMax) {
        return 0.0f;
      }
      const float density = expf(value);
      return density * expf(-density);
    }
    case TextureAlphaActivation::Clamp:
      return value >= 0.0f && value <= 1.0f ? 1.0f : 0.0f;
  }
  return 0.0f;
}

__device__ __forceinline__ float activate_mlp_rgb(float value, int32_t activation) {
  switch (static_cast<MlpRgbActivation>(activation)) {
    case MlpRgbActivation::ShiftedRelu:
      return fmaxf(value + 0.5f, 0.0f);
    case MlpRgbActivation::ShiftedIdentity:
      return value + 0.5f;
    case MlpRgbActivation::Sigmoid:
      return sigmoid(value);
    case MlpRgbActivation::ShiftedClamp:
      return fminf(fmaxf(value + 0.5f, 0.0f), 1.0f);
  }
  return 0.0f;
}

__device__ __forceinline__ float activate_mlp_rgb_backward(float value, int32_t activation) {
  switch (static_cast<MlpRgbActivation>(activation)) {
    case MlpRgbActivation::ShiftedRelu:
      return value > -0.5f ? 1.0f : 0.0f;
    case MlpRgbActivation::ShiftedIdentity:
      return 1.0f;
    case MlpRgbActivation::Sigmoid: {
      const float output = sigmoid(value);
      return output * (1.0f - output);
    }
    case MlpRgbActivation::ShiftedClamp:
      return value >= -0.5f && value <= 0.5f ? 1.0f : 0.0f;
  }
  return 0.0f;
}

__device__ __forceinline__ float activate_mlp_alpha(float value, int32_t activation) {
  switch (static_cast<MlpAlphaActivation>(activation)) {
    case MlpAlphaActivation::ShiftedReluDensity:
      return -expm1f(-fmaxf(value + 0.5f, 0.0f));
    case MlpAlphaActivation::Sigmoid:
      return sigmoid(value);
    case MlpAlphaActivation::ExpDensity: {
      const float bounded_value = value > kExpDensityMax ? kExpDensityMax : value;
      return -expm1f(-expf(bounded_value));
    }
    case MlpAlphaActivation::ShiftedClamp:
      return fminf(fmaxf(value + 0.5f, 0.0f), 1.0f);
  }
  return 0.0f;
}

__device__ __forceinline__ float activate_mlp_alpha_backward(float value, int32_t activation) {
  switch (static_cast<MlpAlphaActivation>(activation)) {
    case MlpAlphaActivation::ShiftedReluDensity:
      return value > -0.5f ? expf(-(value + 0.5f)) : 0.0f;
    case MlpAlphaActivation::Sigmoid: {
      const float output = sigmoid(value);
      return output * (1.0f - output);
    }
    case MlpAlphaActivation::ExpDensity: {
      if (value > kExpDensityMax) {
        return 0.0f;
      }
      const float density = expf(value);
      return density * expf(-density);
    }
    case MlpAlphaActivation::ShiftedClamp:
      return value >= -0.5f && value <= 0.5f ? 1.0f : 0.0f;
  }
  return 0.0f;
}

__device__ __forceinline__ float cross2(float ax, float ay, float bx, float by) {
  return ax * by - ay * bx;
}

__device__ __forceinline__ float
edge_function(float ax, float ay, float bx, float by, float px, float py) {
  const float ex = bx - ax;
  const float ey = by - ay;
  const float dx = px - ax;
  const float dy = py - ay;
  return cross2(ex, ey, dx, dy);
}

__device__ __forceinline__ float canonical_edge_function(
    int32_t vertex_a,
    int32_t vertex_b,
    float ax,
    float ay,
    float bx,
    float by,
    float px,
    float py) {
  if (vertex_a <= vertex_b) {
    return edge_function(ax, ay, bx, by, px, py);
  }
  return -edge_function(bx, by, ax, ay, px, py);
}

__device__ __forceinline__ bool hard_triangle_coverage(
    const TriangleData& triangle,
    float px,
    float py,
    float area,
    float* screen_bary) {
  const float orientation = area >= 0.0f ? 1.0f : -1.0f;
  const float raw_bary[3] = {
      orientation *
          canonical_edge_function(
              triangle.vertex_id[1],
              triangle.vertex_id[2],
              triangle.x[1],
              triangle.y[1],
              triangle.x[2],
              triangle.y[2],
              px,
              py),
      orientation *
          canonical_edge_function(
              triangle.vertex_id[2],
              triangle.vertex_id[0],
              triangle.x[2],
              triangle.y[2],
              triangle.x[0],
              triangle.y[0],
              px,
              py),
      orientation *
          canonical_edge_function(
              triangle.vertex_id[0],
              triangle.vertex_id[1],
              triangle.x[0],
              triangle.y[0],
              triangle.x[1],
              triangle.y[1],
              px,
              py)};
  if (raw_bary[0] < 0.0f || raw_bary[1] < 0.0f || raw_bary[2] < 0.0f) {
    return false;
  }

  const float edge01_x = triangle.x[1] - triangle.x[0];
  const float edge01_y = triangle.y[1] - triangle.y[0];
  const float edge02_x = triangle.x[2] - triangle.x[0];
  const float edge02_y = triangle.y[2] - triangle.y[0];
  const float edge12_x = triangle.x[2] - triangle.x[1];
  const float edge12_y = triangle.y[2] - triangle.y[1];
  const bool positive_winding = area > 0.0f;
  const bool top_left0 = positive_winding
      ? edge12_y < 0.0f || (edge12_y == 0.0f && edge12_x > 0.0f)
      : edge12_y > 0.0f || (edge12_y == 0.0f && edge12_x < 0.0f);
  const bool top_left1 = positive_winding
      ? edge02_y > 0.0f || (edge02_y == 0.0f && edge02_x < 0.0f)
      : edge02_y < 0.0f || (edge02_y == 0.0f && edge02_x > 0.0f);
  const bool top_left2 = positive_winding
      ? edge01_y < 0.0f || (edge01_y == 0.0f && edge01_x > 0.0f)
      : edge01_y > 0.0f || (edge01_y == 0.0f && edge01_x < 0.0f);
  if ((raw_bary[0] == 0.0f && !top_left0) || (raw_bary[1] == 0.0f && !top_left1) ||
      (raw_bary[2] == 0.0f && !top_left2)) {
    return false;
  }

  const float inverse_area = 1.0f / fabsf(area);
#pragma unroll
  for (int32_t corner = 0; corner < 3; ++corner) {
    screen_bary[corner] = raw_bary[corner] * inverse_area;
  }
  return true;
}

__device__ __forceinline__ float load_texture(
    const float* texture,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    int32_t c,
    int32_t y,
    int32_t x) {
  const int32_t batch_index = texture_batch == 1 ? 0 : n;
  const int64_t offset =
      ((static_cast<int64_t>(batch_index) * channels + c) * height + y) * width + x;
  return texture[offset];
}

__device__ __forceinline__ float
compute_padded_coordinate(float coordinate, int32_t size, GridSamplerPadding padding_mode) {
  if (padding_mode == GridSamplerPadding::Border) {
    coordinate = clip_coordinates(coordinate, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    coordinate = reflect_coordinates(coordinate, -1, 2 * size - 1);
    coordinate = clip_coordinates(coordinate, size);
  }
  return safe_downgrade_to_int_range(coordinate);
}

__device__ __forceinline__ float load_texture_bounded(
    const float* texture,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    int32_t c,
    float y,
    float x,
    GridSamplerPadding padding_mode) {
  x = compute_padded_coordinate(x, width, padding_mode);
  y = compute_padded_coordinate(y, height, padding_mode);
  const int32_t ix = static_cast<int32_t>(x);
  const int32_t iy = static_cast<int32_t>(y);
  if (!within_bounds_2d(iy, ix, height, width)) {
    return 0.0f;
  }
  return load_texture(texture, texture_batch, channels, height, width, n, c, iy, ix);
}

__device__ __forceinline__ void add_texture_grad_bounded(
    float* texture_grad,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    int32_t c,
    float y,
    float x,
    GridSamplerPadding padding_mode,
    float value) {
  x = compute_padded_coordinate(x, width, padding_mode);
  y = compute_padded_coordinate(y, height, padding_mode);
  const int32_t ix = static_cast<int32_t>(x);
  const int32_t iy = static_cast<int32_t>(y);
  if (!within_bounds_2d(iy, ix, height, width)) {
    return;
  }
  const int32_t batch_index = texture_batch == 1 ? 0 : n;
  const int64_t offset =
      ((static_cast<int64_t>(batch_index) * channels + c) * height + iy) * width + ix;
  atomicAdd(texture_grad + offset, value);
}

__device__ __forceinline__ void sample_texture_bilinear(
    const float* texture,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    float u,
    float v,
    GridSamplerPadding padding_mode,
    float* output) {
  const float normalized_x = u * 2.0f - 1.0f;
  const float normalized_y = v * 2.0f - 1.0f;
  const float x = grid_sampler_compute_source_index(normalized_x, width, padding_mode, false);
  const float y = grid_sampler_compute_source_index(normalized_y, height, padding_mode, false);
  const int32_t x0 = static_cast<int32_t>(floorf(x));
  const int32_t y0 = static_cast<int32_t>(floorf(y));
  const int32_t x1 = x0 + 1;
  const int32_t y1 = y0 + 1;
  const float wx = x - static_cast<float>(x0);
  const float wy = y - static_cast<float>(y0);

  for (int32_t c = 0; c < channels; ++c) {
    const float value00 = within_bounds_2d(y0, x0, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, y0, x0)
        : 0.0f;
    const float value01 = within_bounds_2d(y0, x1, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, y0, x1)
        : 0.0f;
    const float value10 = within_bounds_2d(y1, x0, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, y1, x0)
        : 0.0f;
    const float value11 = within_bounds_2d(y1, x1, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, y1, x1)
        : 0.0f;
    const float value0 = value00 + wx * (value01 - value00);
    const float value1 = value10 + wx * (value11 - value10);
    output[c] = value0 + wy * (value1 - value0);
  }
}

__device__ __forceinline__ void sample_texture_nearest(
    const float* texture,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    float u,
    float v,
    GridSamplerPadding padding_mode,
    float* output) {
  const float x = grid_sampler_compute_source_index(u * 2.0f - 1.0f, width, padding_mode, false);
  const float y = grid_sampler_compute_source_index(v * 2.0f - 1.0f, height, padding_mode, false);
  const int32_t ix = static_cast<int32_t>(nearbyintf(x));
  const int32_t iy = static_cast<int32_t>(nearbyintf(y));
  for (int32_t c = 0; c < channels; ++c) {
    output[c] = within_bounds_2d(iy, ix, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, iy, ix)
        : 0.0f;
  }
}

__device__ __forceinline__ void sample_texture_bicubic(
    const float* texture,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    float u,
    float v,
    GridSamplerPadding padding_mode,
    float* output) {
  const float x = grid_sampler_unnormalize(u * 2.0f - 1.0f, width, false);
  const float y = grid_sampler_unnormalize(v * 2.0f - 1.0f, height, false);
  const float x0 = floorf(x);
  const float y0 = floorf(y);
  float x_coefficients[4];
  float y_coefficients[4];
  get_cubic_upsampling_coefficients<float>(x_coefficients, x - x0);
  get_cubic_upsampling_coefficients<float>(y_coefficients, y - y0);

  for (int32_t c = 0; c < channels; ++c) {
    float value = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
#pragma unroll
      for (int32_t i = 0; i < 4; ++i) {
        value += load_texture_bounded(
                     texture,
                     texture_batch,
                     channels,
                     height,
                     width,
                     n,
                     c,
                     y0 - 1.0f + static_cast<float>(j),
                     x0 - 1.0f + static_cast<float>(i),
                     padding_mode) *
            x_coefficients[i] * y_coefficients[j];
      }
    }
    output[c] = value;
  }
}

__device__ __forceinline__ void sample_texture(
    const float* texture,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    float u,
    float v,
    int32_t interpolation_mode,
    int32_t padding_mode,
    float* output) {
  const auto padding = static_cast<GridSamplerPadding>(padding_mode);
  switch (static_cast<TextureInterpolation>(interpolation_mode)) {
    case TextureInterpolation::Bilinear:
      sample_texture_bilinear(
          texture, texture_batch, channels, height, width, n, u, v, padding, output);
      break;
    case TextureInterpolation::Nearest:
      sample_texture_nearest(
          texture, texture_batch, channels, height, width, n, u, v, padding, output);
      break;
    case TextureInterpolation::Bicubic:
      sample_texture_bicubic(
          texture, texture_batch, channels, height, width, n, u, v, padding, output);
      break;
  }
}

__device__ __forceinline__ void sample_texture_bilinear_backward(
    const float* texture,
    float* texture_grad,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    float u,
    float v,
    GridSamplerPadding padding_mode,
    const float* output_grad,
    float& u_grad,
    float& v_grad) {
  const int32_t batch_index = texture_batch == 1 ? 0 : n;
  float x_multiplier = 0.0f;
  float y_multiplier = 0.0f;
  const float x = grid_sampler_compute_source_index_set_grad(
      u * 2.0f - 1.0f, width, padding_mode, false, &x_multiplier);
  const float y = grid_sampler_compute_source_index_set_grad(
      v * 2.0f - 1.0f, height, padding_mode, false, &y_multiplier);
  const int32_t x0 = static_cast<int32_t>(floorf(x));
  const int32_t y0 = static_cast<int32_t>(floorf(y));
  const int32_t x1 = x0 + 1;
  const int32_t y1 = y0 + 1;
  const float wx = x - static_cast<float>(x0);
  const float wy = y - static_cast<float>(y0);
  const float weight00 = (1.0f - wx) * (1.0f - wy);
  const float weight01 = wx * (1.0f - wy);
  const float weight10 = (1.0f - wx) * wy;
  const float weight11 = wx * wy;

  float x_grad = 0.0f;
  float y_grad = 0.0f;
  for (int32_t c = 0; c < channels; ++c) {
    const float value00 = within_bounds_2d(y0, x0, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, y0, x0)
        : 0.0f;
    const float value01 = within_bounds_2d(y0, x1, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, y0, x1)
        : 0.0f;
    const float value10 = within_bounds_2d(y1, x0, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, y1, x0)
        : 0.0f;
    const float value11 = within_bounds_2d(y1, x1, height, width)
        ? load_texture(texture, texture_batch, channels, height, width, n, c, y1, x1)
        : 0.0f;
    const float grad = output_grad[c];
    x_grad += grad * ((value01 - value00) * (1.0f - wy) + (value11 - value10) * wy);
    y_grad += grad * ((value10 - value00) * (1.0f - wx) + (value11 - value01) * wx);

    const int64_t base = static_cast<int64_t>(batch_index) * channels + c;
    if (within_bounds_2d(y0, x0, height, width)) {
      atomicAdd(texture_grad + (base * height + y0) * width + x0, grad * weight00);
    }
    if (within_bounds_2d(y0, x1, height, width)) {
      atomicAdd(texture_grad + (base * height + y0) * width + x1, grad * weight01);
    }
    if (within_bounds_2d(y1, x0, height, width)) {
      atomicAdd(texture_grad + (base * height + y1) * width + x0, grad * weight10);
    }
    if (within_bounds_2d(y1, x1, height, width)) {
      atomicAdd(texture_grad + (base * height + y1) * width + x1, grad * weight11);
    }
  }
  u_grad = x_grad * x_multiplier * 2.0f;
  v_grad = y_grad * y_multiplier * 2.0f;
}

__device__ __forceinline__ void sample_texture_nearest_backward(
    float* texture_grad,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    float u,
    float v,
    GridSamplerPadding padding_mode,
    const float* output_grad,
    float& u_grad,
    float& v_grad) {
  const float x = grid_sampler_compute_source_index(u * 2.0f - 1.0f, width, padding_mode, false);
  const float y = grid_sampler_compute_source_index(v * 2.0f - 1.0f, height, padding_mode, false);
  const int32_t ix = static_cast<int32_t>(nearbyintf(x));
  const int32_t iy = static_cast<int32_t>(nearbyintf(y));
  if (within_bounds_2d(iy, ix, height, width)) {
    const int32_t batch_index = texture_batch == 1 ? 0 : n;
    for (int32_t c = 0; c < channels; ++c) {
      const int64_t offset =
          ((static_cast<int64_t>(batch_index) * channels + c) * height + iy) * width + ix;
      atomicAdd(texture_grad + offset, output_grad[c]);
    }
  }
  u_grad = 0.0f;
  v_grad = 0.0f;
}

__device__ __forceinline__ void get_cubic_coefficients_grad(float coefficients[4], float t) {
  constexpr float kCubicCoefficient = -0.75f;
  float x = -1.0f - t;
  coefficients[0] =
      (-3.0f * kCubicCoefficient * x - 10.0f * kCubicCoefficient) * x - 8.0f * kCubicCoefficient;
  x = -t;
  coefficients[1] =
      (-3.0f * (kCubicCoefficient + 2.0f) * x - 2.0f * (kCubicCoefficient + 3.0f)) * x;
  x = 1.0f - t;
  coefficients[2] = (3.0f * (kCubicCoefficient + 2.0f) * x - 2.0f * (kCubicCoefficient + 3.0f)) * x;
  x = 2.0f - t;
  coefficients[3] =
      (3.0f * kCubicCoefficient * x - 10.0f * kCubicCoefficient) * x + 8.0f * kCubicCoefficient;
}

__device__ __forceinline__ void sample_texture_bicubic_backward(
    const float* texture,
    float* texture_grad,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    float u,
    float v,
    GridSamplerPadding padding_mode,
    const float* output_grad,
    float& u_grad,
    float& v_grad) {
  float x_multiplier = 0.0f;
  float y_multiplier = 0.0f;
  const float x = grid_sampler_unnormalize_set_grad(u * 2.0f - 1.0f, width, false, &x_multiplier);
  const float y = grid_sampler_unnormalize_set_grad(v * 2.0f - 1.0f, height, false, &y_multiplier);
  const float x0 = floorf(x);
  const float y0 = floorf(y);
  float x_coefficients[4];
  float y_coefficients[4];
  float x_coefficient_grads[4];
  float y_coefficient_grads[4];
  get_cubic_upsampling_coefficients<float>(x_coefficients, x - x0);
  get_cubic_upsampling_coefficients<float>(y_coefficients, y - y0);
  get_cubic_coefficients_grad(x_coefficient_grads, x - x0);
  get_cubic_coefficients_grad(y_coefficient_grads, y - y0);

  float x_grad = 0.0f;
  float y_grad = 0.0f;
  for (int32_t c = 0; c < channels; ++c) {
    const float grad = output_grad[c];
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
#pragma unroll
      for (int32_t i = 0; i < 4; ++i) {
        const float sample_x = x0 - 1.0f + static_cast<float>(i);
        const float sample_y = y0 - 1.0f + static_cast<float>(j);
        add_texture_grad_bounded(
            texture_grad,
            texture_batch,
            channels,
            height,
            width,
            n,
            c,
            sample_y,
            sample_x,
            padding_mode,
            grad * x_coefficients[i] * y_coefficients[j]);
        const float value = load_texture_bounded(
            texture,
            texture_batch,
            channels,
            height,
            width,
            n,
            c,
            sample_y,
            sample_x,
            padding_mode);
        x_grad -= grad * value * x_coefficient_grads[i] * y_coefficients[j];
        y_grad -= grad * value * y_coefficient_grads[j] * x_coefficients[i];
      }
    }
  }
  u_grad = x_grad * x_multiplier * 2.0f;
  v_grad = y_grad * y_multiplier * 2.0f;
}

__device__ __forceinline__ void sample_texture_backward(
    const float* texture,
    float* texture_grad,
    int32_t texture_batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t n,
    float u,
    float v,
    int32_t interpolation_mode,
    int32_t padding_mode,
    const float* output_grad,
    float& u_grad,
    float& v_grad) {
  const auto padding = static_cast<GridSamplerPadding>(padding_mode);
  switch (static_cast<TextureInterpolation>(interpolation_mode)) {
    case TextureInterpolation::Bilinear:
      sample_texture_bilinear_backward(
          texture,
          texture_grad,
          texture_batch,
          channels,
          height,
          width,
          n,
          u,
          v,
          padding,
          output_grad,
          u_grad,
          v_grad);
      break;
    case TextureInterpolation::Nearest:
      sample_texture_nearest_backward(
          texture_grad,
          texture_batch,
          channels,
          height,
          width,
          n,
          u,
          v,
          padding,
          output_grad,
          u_grad,
          v_grad);
      break;
    case TextureInterpolation::Bicubic:
      sample_texture_bicubic_backward(
          texture,
          texture_grad,
          texture_batch,
          channels,
          height,
          width,
          n,
          u,
          v,
          padding,
          output_grad,
          u_grad,
          v_grad);
      break;
  }
}

__device__ __forceinline__ void perspective_barycentric_backward(
    const TriangleData& triangle,
    float px,
    float py,
    const float* screen_bary,
    const float* bary,
    float inverse_depth,
    const float* bary_grad,
    float* x_grad,
    float* y_grad,
    float* z_grad) {
  const float bary_dot_grad =
      bary[0] * bary_grad[0] + bary[1] * bary_grad[1] + bary[2] * bary_grad[2];
  float screen_bary_grad[3];
#pragma unroll
  for (int32_t corner = 0; corner < 3; ++corner) {
    const float q_grad = (bary_grad[corner] - bary_dot_grad) / inverse_depth;
    const float z = clamp_away_from_zero(triangle.z[corner]);
    screen_bary_grad[corner] = q_grad / z;
    z_grad[corner] += -q_grad * screen_bary[corner] / (z * z);
  }

  const float edge01_x = triangle.x[1] - triangle.x[0];
  const float edge01_y = triangle.y[1] - triangle.y[0];
  const float edge02_x = triangle.x[2] - triangle.x[0];
  const float edge02_y = triangle.y[2] - triangle.y[0];
  const float delta_x = px - triangle.x[0];
  const float delta_y = py - triangle.y[0];
  const float area = cross2(edge01_x, edge01_y, edge02_x, edge02_y);
  const float bary12_grad_x = -screen_bary_grad[0] + screen_bary_grad[1];
  const float bary12_grad_y = -screen_bary_grad[0] + screen_bary_grad[2];
  const float pre_grad_x = bary12_grad_x / area;
  const float pre_grad_y = bary12_grad_y / area;
  const float area_grad = -(pre_grad_x * screen_bary[1] + pre_grad_y * screen_bary[2]);
  const float delta_grad_x = pre_grad_x * edge02_y - pre_grad_y * edge01_y;
  const float delta_grad_y = -pre_grad_x * edge02_x + pre_grad_y * edge01_x;
  const float edge02_grad_x = -pre_grad_x * delta_y - area_grad * edge01_y;
  const float edge02_grad_y = pre_grad_x * delta_x + area_grad * edge01_x;
  const float edge01_grad_x = pre_grad_y * delta_y + area_grad * edge02_y;
  const float edge01_grad_y = -pre_grad_y * delta_x - area_grad * edge02_x;

  x_grad[0] += -edge02_grad_x - edge01_grad_x - delta_grad_x;
  y_grad[0] += -edge02_grad_y - edge01_grad_y - delta_grad_y;
  x_grad[1] += edge01_grad_x;
  y_grad[1] += edge01_grad_y;
  x_grad[2] += edge02_grad_x;
  y_grad[2] += edge02_grad_y;
}

template <int LayerCount>
__device__ __forceinline__ void
evaluate_tiny_mlp(const TinyMlpParams& mlp, const float* features, float* rgba) {
  static_assert(LayerCount == 4 || LayerCount == 5);
  float current[kMlpWidth];
  float next[kMlpWidth];

#pragma unroll
  for (int32_t output_channel = 0; output_channel < kMlpWidth; ++output_channel) {
    float value = mlp.biases[0][output_channel];
    const float* weight = mlp.weights[0] + output_channel * mlp.input_channels;
    for (int32_t input_channel = 0; input_channel < mlp.input_channels; ++input_channel) {
      value += weight[input_channel] * features[input_channel];
    }
    current[output_channel] = fmaxf(value, value * mlp.negative_slope);
  }

#pragma unroll
  for (int32_t layer = 1; layer < LayerCount - 1; ++layer) {
#pragma unroll
    for (int32_t output_channel = 0; output_channel < kMlpWidth; ++output_channel) {
      float value = mlp.biases[layer][output_channel];
      const float* weight = mlp.weights[layer] + output_channel * kMlpWidth;
#pragma unroll
      for (int32_t input_channel = 0; input_channel < kMlpWidth; ++input_channel) {
        value += weight[input_channel] * current[input_channel];
      }
      next[output_channel] = fmaxf(value, value * mlp.negative_slope);
    }
#pragma unroll
    for (int32_t channel = 0; channel < kMlpWidth; ++channel) {
      current[channel] = next[channel];
    }
  }

  constexpr int32_t kOutputLayer = LayerCount - 1;
#pragma unroll
  for (int32_t output_channel = 0; output_channel < 4; ++output_channel) {
    float value = mlp.biases[kOutputLayer][output_channel];
    const float* weight = mlp.weights[kOutputLayer] + output_channel * kMlpWidth;
#pragma unroll
    for (int32_t input_channel = 0; input_channel < kMlpWidth; ++input_channel) {
      value += weight[input_channel] * current[input_channel];
    }
    rgba[output_channel] = output_channel == 3 ? activate_mlp_alpha(value, mlp.alpha_activation)
                                               : activate_mlp_rgb(value, mlp.rgb_activation);
  }
}

template <int LayerCount>
__device__ __forceinline__ void evaluate_tiny_mlp_backward(
    const TinyMlpParams& mlp,
    const TinyMlpGradParams& mlp_grad,
    const float* features,
    const float* rgba_grad,
    float* feature_grad) {
  static_assert(LayerCount == 4 || LayerCount == 5);
  float preactivation[LayerCount - 1][kMlpWidth];
  float activation[LayerCount - 1][kMlpWidth];

#pragma unroll
  for (int32_t output_channel = 0; output_channel < kMlpWidth; ++output_channel) {
    float value = mlp.biases[0][output_channel];
    const float* weight = mlp.weights[0] + output_channel * mlp.input_channels;
    for (int32_t input_channel = 0; input_channel < mlp.input_channels; ++input_channel) {
      value += weight[input_channel] * features[input_channel];
    }
    preactivation[0][output_channel] = value;
    activation[0][output_channel] = fmaxf(value, value * mlp.negative_slope);
  }

#pragma unroll
  for (int32_t layer = 1; layer < LayerCount - 1; ++layer) {
#pragma unroll
    for (int32_t output_channel = 0; output_channel < kMlpWidth; ++output_channel) {
      float value = mlp.biases[layer][output_channel];
      const float* weight = mlp.weights[layer] + output_channel * kMlpWidth;
#pragma unroll
      for (int32_t input_channel = 0; input_channel < kMlpWidth; ++input_channel) {
        value += weight[input_channel] * activation[layer - 1][input_channel];
      }
      preactivation[layer][output_channel] = value;
      activation[layer][output_channel] = fmaxf(value, value * mlp.negative_slope);
    }
  }

  float activation_grad[kMlpWidth] = {0.0f};
  constexpr int32_t kOutputLayer = LayerCount - 1;
#pragma unroll
  for (int32_t output_channel = 0; output_channel < 4; ++output_channel) {
    const float* weight = mlp.weights[kOutputLayer] + output_channel * kMlpWidth;
    float value = mlp.biases[kOutputLayer][output_channel];
#pragma unroll
    for (int32_t input_channel = 0; input_channel < kMlpWidth; ++input_channel) {
      value += weight[input_channel] * activation[LayerCount - 2][input_channel];
    }
    float grad = rgba_grad[output_channel] *
        (output_channel == 3 ? activate_mlp_alpha_backward(value, mlp.alpha_activation)
                             : activate_mlp_rgb_backward(value, mlp.rgb_activation));
    atomicAdd(mlp_grad.biases[kOutputLayer] + output_channel, grad);
    float* weight_grad = mlp_grad.weights[kOutputLayer] + output_channel * kMlpWidth;
#pragma unroll
    for (int32_t input_channel = 0; input_channel < kMlpWidth; ++input_channel) {
      atomicAdd(weight_grad + input_channel, grad * activation[LayerCount - 2][input_channel]);
      activation_grad[input_channel] += grad * weight[input_channel];
    }
  }

  for (int32_t layer = LayerCount - 2; layer >= 0; --layer) {
    float previous_grad[kMlpWidth] = {0.0f};
#pragma unroll
    for (int32_t output_channel = 0; output_channel < kMlpWidth; ++output_channel) {
      const float activation_derivative =
          preactivation[layer][output_channel] > 0.0f ? 1.0f : mlp.negative_slope;
      const float grad = activation_grad[output_channel] * activation_derivative;
      atomicAdd(mlp_grad.biases[layer] + output_channel, grad);
      const int32_t input_channels = layer == 0 ? mlp.input_channels : kMlpWidth;
      const float* input = layer == 0 ? features : activation[layer - 1];
      const float* weight = mlp.weights[layer] + output_channel * input_channels;
      float* weight_grad = mlp_grad.weights[layer] + output_channel * input_channels;
      for (int32_t input_channel = 0; input_channel < input_channels; ++input_channel) {
        atomicAdd(weight_grad + input_channel, grad * input[input_channel]);
        if (layer == 0) {
          feature_grad[input_channel] += grad * weight[input_channel];
        } else {
          previous_grad[input_channel] += grad * weight[input_channel];
        }
      }
    }
    if (layer > 0) {
#pragma unroll
      for (int32_t channel = 0; channel < kMlpWidth; ++channel) {
        activation_grad[channel] = previous_grad[channel];
      }
    }
  }
}

struct TextureFragmentEvaluator {
  const float* texture;
  int32_t texture_batch;
  int32_t channels;
  int32_t height;
  int32_t width;
  int32_t interpolation_mode;
  int32_t padding_mode;
  int32_t rgb_activation;
  int32_t alpha_activation;

  __device__ __forceinline__ void evaluate(
      int32_t n,
      const TriangleData& triangle,
      const float* bary,
      float* rgb,
      float& alpha) const {
    float u = 0.0f;
    float v = 0.0f;
#pragma unroll
    for (int32_t corner = 0; corner < 3; ++corner) {
      u += bary[corner] * triangle.u[corner];
      v += bary[corner] * triangle.v[corner];
    }
    float texel[4];
    sample_texture(
        texture,
        texture_batch,
        channels,
        height,
        width,
        n,
        u,
        v,
        interpolation_mode,
        padding_mode,
        texel);
    rgb[0] = activate_texture_rgb(texel[0], rgb_activation);
    rgb[1] = activate_texture_rgb(texel[1], rgb_activation);
    rgb[2] = activate_texture_rgb(texel[2], rgb_activation);
    alpha = channels == 4 ? activate_texture_alpha(texel[3], alpha_activation) : 1.0f;
  }
};

struct TextureFragmentBackward {
  TextureFragmentEvaluator forward;
  float* texture_grad;

  __device__ __forceinline__ int32_t shared_weight_grad_count() const {
    return 0;
  }

  __device__ __forceinline__ int32_t shared_bias_grad_count() const {
    return 0;
  }

  __device__ __forceinline__ void bind_shared_mlp_grads(float*, float*) {}

  __device__ __forceinline__ void flush_shared_mlp_grads(int32_t, int32_t) const {}

  __device__ __forceinline__ void evaluate(
      int32_t n,
      const TriangleData& triangle,
      const float* bary,
      float* rgb,
      float& alpha) const {
    forward.evaluate(n, triangle, bary, rgb, alpha);
  }

  __device__ __forceinline__ void backward(
      int32_t n,
      const TriangleData& triangle,
      const float* bary,
      const float* rgb_grad,
      float alpha_grad,
      float /*appearance_alpha*/,
      FragmentAttributeGrads& attribute_grad) const {
    float u = 0.0f;
    float v = 0.0f;
#pragma unroll
    for (int32_t corner = 0; corner < 3; ++corner) {
      u += bary[corner] * triangle.u[corner];
      v += bary[corner] * triangle.v[corner];
    }
    float texel[4];
    sample_texture(
        forward.texture,
        forward.texture_batch,
        forward.channels,
        forward.height,
        forward.width,
        n,
        u,
        v,
        forward.interpolation_mode,
        forward.padding_mode,
        texel);
    const float texel_grad[4] = {
        rgb_grad[0] * activate_texture_rgb_backward(texel[0], forward.rgb_activation),
        rgb_grad[1] * activate_texture_rgb_backward(texel[1], forward.rgb_activation),
        rgb_grad[2] * activate_texture_rgb_backward(texel[2], forward.rgb_activation),
        forward.channels == 4
            ? alpha_grad * activate_texture_alpha_backward(texel[3], forward.alpha_activation)
            : 0.0f};
    float u_grad = 0.0f;
    float v_grad = 0.0f;
    sample_texture_backward(
        forward.texture,
        texture_grad,
        forward.texture_batch,
        forward.channels,
        forward.height,
        forward.width,
        n,
        u,
        v,
        forward.interpolation_mode,
        forward.padding_mode,
        texel_grad,
        u_grad,
        v_grad);
#pragma unroll
    for (int32_t corner = 0; corner < 3; ++corner) {
      attribute_grad.u[corner] += bary[corner] * u_grad;
      attribute_grad.v[corner] += bary[corner] * v_grad;
      attribute_grad.bary[corner] += u_grad * triangle.u[corner] + v_grad * triangle.v[corner];
    }
  }
};

template <int LayerCount>
struct MlpFragmentEvaluator {
  const float* texture;
  const float* reference_texture;
  int32_t texture_batch;
  int32_t reference_texture_batch;
  int32_t texture_channels;
  int32_t reference_texture_channels;
  int32_t texture_height;
  int32_t texture_width;
  int32_t reference_texture_height;
  int32_t reference_texture_width;
  int32_t interpolation_mode;
  int32_t padding_mode;
  TinyMlpParams mlp;

  __device__ __forceinline__ void evaluate(
      int32_t n,
      const TriangleData& triangle,
      const float* bary,
      float* rgb,
      float& alpha) const {
    float u = 0.0f;
    float v = 0.0f;
    float reference_u = 0.0f;
    float reference_v = 0.0f;
#pragma unroll
    for (int32_t corner = 0; corner < 3; ++corner) {
      u += bary[corner] * triangle.u[corner];
      v += bary[corner] * triangle.v[corner];
      reference_u += bary[corner] * triangle.reference_u[corner];
      reference_v += bary[corner] * triangle.reference_v[corner];
    }

    float features[kMaxFeatureChannels];
    sample_texture(
        texture,
        texture_batch,
        texture_channels,
        texture_height,
        texture_width,
        n,
        u,
        v,
        interpolation_mode,
        padding_mode,
        features);
    sample_texture(
        reference_texture,
        reference_texture_batch,
        reference_texture_channels,
        reference_texture_height,
        reference_texture_width,
        n,
        reference_u,
        reference_v,
        interpolation_mode,
        padding_mode,
        features + texture_channels);
    float rgba[4];
    evaluate_tiny_mlp<LayerCount>(mlp, features, rgba);
    rgb[0] = rgba[0];
    rgb[1] = rgba[1];
    rgb[2] = rgba[2];
    alpha = rgba[3];
  }
};

template <int LayerCount>
struct MlpFragmentBackward {
  MlpFragmentEvaluator<LayerCount> forward;
  float* texture_grad;
  float* reference_texture_grad;
  TinyMlpGradParams mlp_grad;
  TinyMlpGradParams global_mlp_grad;

  __device__ __forceinline__ int32_t shared_weight_grad_count() const {
    return kMlpWidth * forward.mlp.input_channels + (LayerCount - 2) * kMlpWidth * kMlpWidth +
        4 * kMlpWidth;
  }

  __device__ __forceinline__ int32_t shared_bias_grad_count() const {
    return (LayerCount - 1) * kMlpWidth + 4;
  }

  __device__ __forceinline__ void bind_shared_mlp_grads(
      float* shared_weight_grad,
      float* shared_bias_grad) {
    global_mlp_grad = mlp_grad;
    int32_t weight_offset = 0;
    int32_t bias_offset = 0;
#pragma unroll
    for (int32_t layer = 0; layer < LayerCount; ++layer) {
      const int32_t input_channels = layer == 0 ? forward.mlp.input_channels : kMlpWidth;
      const int32_t output_channels = layer == LayerCount - 1 ? 4 : kMlpWidth;
      mlp_grad.weights[layer] = shared_weight_grad + weight_offset;
      mlp_grad.biases[layer] = shared_bias_grad + bias_offset;
      weight_offset += input_channels * output_channels;
      bias_offset += output_channels;
    }
  }

  __device__ __forceinline__ void flush_shared_mlp_grads(int32_t thread_index, int32_t thread_count)
      const {
#pragma unroll
    for (int32_t layer = 0; layer < LayerCount; ++layer) {
      const int32_t input_channels = layer == 0 ? forward.mlp.input_channels : kMlpWidth;
      const int32_t output_channels = layer == LayerCount - 1 ? 4 : kMlpWidth;
      const int32_t weight_count = input_channels * output_channels;
      for (int32_t index = thread_index; index < weight_count; index += thread_count) {
        atomicAdd(global_mlp_grad.weights[layer] + index, mlp_grad.weights[layer][index]);
      }
      for (int32_t index = thread_index; index < output_channels; index += thread_count) {
        atomicAdd(global_mlp_grad.biases[layer] + index, mlp_grad.biases[layer][index]);
      }
    }
  }

  __device__ __forceinline__ void evaluate(
      int32_t n,
      const TriangleData& triangle,
      const float* bary,
      float* rgb,
      float& alpha) const {
    forward.evaluate(n, triangle, bary, rgb, alpha);
  }

  __device__ __forceinline__ void backward(
      int32_t n,
      const TriangleData& triangle,
      const float* bary,
      const float* rgb_grad,
      float alpha_grad,
      float /*appearance_alpha*/,
      FragmentAttributeGrads& attribute_grad) const {
    float u = 0.0f;
    float v = 0.0f;
    float reference_u = 0.0f;
    float reference_v = 0.0f;
#pragma unroll
    for (int32_t corner = 0; corner < 3; ++corner) {
      u += bary[corner] * triangle.u[corner];
      v += bary[corner] * triangle.v[corner];
      reference_u += bary[corner] * triangle.reference_u[corner];
      reference_v += bary[corner] * triangle.reference_v[corner];
    }
    float features[kMaxFeatureChannels];
    sample_texture(
        forward.texture,
        forward.texture_batch,
        forward.texture_channels,
        forward.texture_height,
        forward.texture_width,
        n,
        u,
        v,
        forward.interpolation_mode,
        forward.padding_mode,
        features);
    sample_texture(
        forward.reference_texture,
        forward.reference_texture_batch,
        forward.reference_texture_channels,
        forward.reference_texture_height,
        forward.reference_texture_width,
        n,
        reference_u,
        reference_v,
        forward.interpolation_mode,
        forward.padding_mode,
        features + forward.texture_channels);
    float feature_grad[kMaxFeatureChannels] = {0.0f};
    const float rgba_grad[4] = {rgb_grad[0], rgb_grad[1], rgb_grad[2], alpha_grad};
    evaluate_tiny_mlp_backward<LayerCount>(
        forward.mlp, mlp_grad, features, rgba_grad, feature_grad);

    float u_grad = 0.0f;
    float v_grad = 0.0f;
    sample_texture_backward(
        forward.texture,
        texture_grad,
        forward.texture_batch,
        forward.texture_channels,
        forward.texture_height,
        forward.texture_width,
        n,
        u,
        v,
        forward.interpolation_mode,
        forward.padding_mode,
        feature_grad,
        u_grad,
        v_grad);
    float reference_u_grad = 0.0f;
    float reference_v_grad = 0.0f;
    sample_texture_backward(
        forward.reference_texture,
        reference_texture_grad,
        forward.reference_texture_batch,
        forward.reference_texture_channels,
        forward.reference_texture_height,
        forward.reference_texture_width,
        n,
        reference_u,
        reference_v,
        forward.interpolation_mode,
        forward.padding_mode,
        feature_grad + forward.texture_channels,
        reference_u_grad,
        reference_v_grad);
#pragma unroll
    for (int32_t corner = 0; corner < 3; ++corner) {
      attribute_grad.u[corner] += bary[corner] * u_grad;
      attribute_grad.v[corner] += bary[corner] * v_grad;
      attribute_grad.reference_u[corner] += bary[corner] * reference_u_grad;
      attribute_grad.reference_v[corner] += bary[corner] * reference_v_grad;
      attribute_grad.bary[corner] += u_grad * triangle.u[corner] + v_grad * triangle.v[corner] +
          reference_u_grad * triangle.reference_u[corner] +
          reference_v_grad * triangle.reference_v[corner];
    }
  }
};

__global__ void compute_tile_bounds_kernel(
    int64_t face_element_count,
    const float* vertices,
    const int32_t* faces,
    int32_t vertex_count,
    int32_t face_count,
    int32_t image_height,
    int32_t image_width,
    int32_t tile_height,
    int32_t tile_width,
    int32_t* tile_bounds,
    int32_t* tile_counts,
    float* face_depths) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= face_element_count) {
    return;
  }
  const int32_t n = static_cast<int32_t>(index / face_count);
  const int32_t f = static_cast<int32_t>(index % face_count);
  const int32_t* face = faces + (static_cast<int64_t>(n) * face_count + f) * 3;
  const int32_t i0 = face[0];
  const int32_t i1 = face[1];
  const int32_t i2 = face[2];
  const bool valid_indices =
      i0 >= 0 && i0 < vertex_count && i1 >= 0 && i1 < vertex_count && i2 >= 0 && i2 < vertex_count;
  if (!valid_indices) {
    tile_bounds[index * 4 + 0] = 0;
    tile_bounds[index * 4 + 1] = 0;
    tile_bounds[index * 4 + 2] = 0;
    tile_bounds[index * 4 + 3] = 0;
    tile_counts[index] = 0;
    face_depths[index] = 0.0f;
    return;
  }
  const float* vertex_batch = vertices + static_cast<int64_t>(n) * vertex_count * 3;
  const float* p0 = vertex_batch + static_cast<int64_t>(i0) * 3;
  const float* p1 = vertex_batch + static_cast<int64_t>(i1) * 3;
  const float* p2 = vertex_batch + static_cast<int64_t>(i2) * 3;

  const float min_x = fminf(fminf(p0[0], p1[0]), p2[0]);
  const float min_y = fminf(fminf(p0[1], p1[1]), p2[1]);
  const float max_x = fmaxf(fmaxf(p0[0], p1[0]), p2[0]);
  const float max_y = fmaxf(fmaxf(p0[1], p1[1]), p2[1]);
  const float area = cross2(p1[0] - p0[0], p1[1] - p0[1], p2[0] - p0[0], p2[1] - p0[1]);
  const float depth = (p0[2] + p1[2] + p2[2]) / 3.0f;
  face_depths[index] = depth;

  int32_t min_tile_x = 0;
  int32_t min_tile_y = 0;
  int32_t max_tile_x = 0;
  int32_t max_tile_y = 0;
  const bool finite_vertices = isfinite(p0[0]) && isfinite(p0[1]) && isfinite(p0[2]) &&
      isfinite(p1[0]) && isfinite(p1[1]) && isfinite(p1[2]) && isfinite(p2[0]) && isfinite(p2[1]) &&
      isfinite(p2[2]);
  const bool valid = finite_vertices && i0 != i1 && i1 != i2 && i2 != i0 &&
      fabsf(area) > 1.0e-12f && p0[2] > 1.0e-8f && p1[2] > 1.0e-8f && p2[2] > 1.0e-8f &&
      max_x >= 0.0f && max_y >= 0.0f && min_x <= static_cast<float>(image_width - 1) &&
      min_y <= static_cast<float>(image_height - 1);
  if (valid) {
    min_tile_x = max(0, static_cast<int32_t>(floorf(min_x / kTileSize)));
    min_tile_y = max(0, static_cast<int32_t>(floorf(min_y / kTileSize)));
    max_tile_x = min(tile_width, static_cast<int32_t>(floorf(max_x / kTileSize)) + 1);
    max_tile_y = min(tile_height, static_cast<int32_t>(floorf(max_y / kTileSize)) + 1);
  }

  tile_bounds[index * 4 + 0] = min_tile_x;
  tile_bounds[index * 4 + 1] = min_tile_y;
  tile_bounds[index * 4 + 2] = max_tile_x;
  tile_bounds[index * 4 + 3] = max_tile_y;
  tile_counts[index] = (max_tile_x - min_tile_x) * (max_tile_y - min_tile_y);
}

__global__ void emit_tile_keys_kernel(
    int64_t face_element_count,
    int32_t face_count,
    int32_t tile_width,
    int32_t tiles_per_image,
    const int32_t* tile_bounds,
    const int64_t* cumulative_tile_counts,
    const float* face_depths,
    uint64_t* keys,
    int32_t* face_ids) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= face_element_count) {
    return;
  }
  const int32_t n = static_cast<int32_t>(index / face_count);
  const int32_t min_tile_x = tile_bounds[index * 4 + 0];
  const int32_t min_tile_y = tile_bounds[index * 4 + 1];
  const int32_t max_tile_x = tile_bounds[index * 4 + 2];
  const int32_t max_tile_y = tile_bounds[index * 4 + 3];
  int64_t output = index == 0 ? 0 : cumulative_tile_counts[index - 1];
  const uint32_t depth_bits = __float_as_uint(face_depths[index]);
  for (int32_t tile_y = min_tile_y; tile_y < max_tile_y; ++tile_y) {
    for (int32_t tile_x = min_tile_x; tile_x < max_tile_x; ++tile_x) {
      const uint32_t tile_id =
          static_cast<uint32_t>(n * tiles_per_image + tile_y * tile_width + tile_x);
      keys[output] = (static_cast<uint64_t>(tile_id) << 32) | depth_bits;
      face_ids[output] = static_cast<int32_t>(index);
      ++output;
    }
  }
}

__global__ void find_tile_offsets_kernel(
    int32_t total_tiles,
    int64_t intersection_count,
    const uint64_t* sorted_keys,
    int32_t* offsets) {
  const int32_t tile_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile_id > total_tiles) {
    return;
  }
  const uint64_t target = static_cast<uint64_t>(static_cast<uint32_t>(tile_id)) << 32;
  int64_t first = 0;
  int64_t count = intersection_count;
  while (count > 0) {
    const int64_t step = count / 2;
    const int64_t candidate = first + step;
    if (sorted_keys[candidate] < target) {
      first = candidate + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  offsets[tile_id] = static_cast<int32_t>(first);
}

__device__ __forceinline__ void
load_triangle(const GeometryParams& geometry, int32_t flattened_face_id, TriangleData& triangle) {
  const int32_t n = flattened_face_id / geometry.face_count;
  const int32_t f = flattened_face_id % geometry.face_count;
  const int32_t* face = geometry.faces + (static_cast<int64_t>(n) * geometry.face_count + f) * 3;
  const float* vertex_batch =
      geometry.vertices + static_cast<int64_t>(n) * geometry.vertex_count * 3;
  const float* uv_batch = geometry.uv + static_cast<int64_t>(n) * geometry.vertex_count * 2;
  const float* reference_uv_batch = geometry.reference_uv == nullptr
      ? nullptr
      : geometry.reference_uv + static_cast<int64_t>(n) * geometry.vertex_count * 2;
  const float* opacity_batch = geometry.opacity + static_cast<int64_t>(n) * geometry.vertex_count;
#pragma unroll
  for (int32_t corner = 0; corner < 3; ++corner) {
    const int32_t vertex_id = face[corner];
    const float* vertex = vertex_batch + static_cast<int64_t>(vertex_id) * 3;
    const float* uv = uv_batch + static_cast<int64_t>(vertex_id) * 2;
    triangle.vertex_id[corner] = vertex_id;
    triangle.x[corner] = vertex[0];
    triangle.y[corner] = vertex[1];
    triangle.z[corner] = vertex[2];
    triangle.u[corner] = uv[0];
    triangle.v[corner] = uv[1];
    if (reference_uv_batch != nullptr) {
      const float* reference_uv = reference_uv_batch + static_cast<int64_t>(vertex_id) * 2;
      triangle.reference_u[corner] = reference_uv[0];
      triangle.reference_v[corner] = reference_uv[1];
    } else {
      triangle.reference_u[corner] = 0.0f;
      triangle.reference_v[corner] = 0.0f;
    }
    triangle.opacity[corner] = opacity_batch[vertex_id];
  }
}

// Boundary queries use the same (mean triangle depth, face id) order as the
// forward tile sort. They never introduce per-pixel depth sorting.
__device__ __forceinline__ uint64_t triangle_order_key(const TriangleData& triangle, int32_t face) {
  const float depth = (triangle.z[0] + triangle.z[1] + triangle.z[2]) / 3.0f;
  return (static_cast<uint64_t>(__float_as_uint(depth)) << 32) | static_cast<uint32_t>(face);
}

__global__ void boundary_edge_keys_kernel(
    GeometryParams geometry,
    int32_t edge_count,
    uint64_t* keys,
    int32_t* edge_ids) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= edge_count) {
    return;
  }
  edge_ids[index] = index;
  keys[index] = UINT64_MAX;
  const int32_t face = index / 3;
  const int32_t n = face / geometry.face_count;
  const int32_t* indices = geometry.faces + static_cast<int64_t>(face) * 3;
#pragma unroll
  for (int32_t c = 0; c < 3; ++c) {
    if (indices[c] < 0 || indices[c] >= geometry.vertex_count) {
      return;
    }
  }
  TriangleData triangle;
  load_triangle(geometry, face, triangle);
#pragma unroll
  for (int32_t c = 0; c < 3; ++c) {
    if (!isfinite(triangle.x[c]) || !isfinite(triangle.y[c]) || !isfinite(triangle.z[c]) ||
        triangle.z[c] <= 1.0e-8f) {
      return;
    }
  }
  const float area = cross2(
      triangle.x[1] - triangle.x[0],
      triangle.y[1] - triangle.y[0],
      triangle.x[2] - triangle.x[0],
      triangle.y[2] - triangle.y[0]);
  if (fabsf(area) <= 1.0e-12f) {
    return;
  }
  const int32_t c = index % 3;
  const uint32_t a =
      static_cast<uint32_t>(n * static_cast<int64_t>(geometry.vertex_count) + indices[c]);
  const uint32_t b =
      static_cast<uint32_t>(n * static_cast<int64_t>(geometry.vertex_count) + indices[(c + 1) % 3]);
  keys[index] = (static_cast<uint64_t>(min(a, b)) << 32) | max(a, b);
}

__device__ __forceinline__ float
edge_side(const TriangleData& triangle, int32_t edge, int32_t first_vertex) {
  const int32_t a = triangle.vertex_id[edge] == first_vertex ? edge : (edge + 1) % 3;
  const int32_t b = a == edge ? (edge + 1) % 3 : edge;
  const int32_t c = (edge + 2) % 3;
  return cross2(
      triangle.x[b] - triangle.x[a],
      triangle.y[b] - triangle.y[a],
      triangle.x[c] - triangle.x[a],
      triangle.y[c] - triangle.y[a]);
}

// Indexed manifold edges have one or two incident faces; the loop also
// handles nonmanifold bundles without a per-edge fragment buffer.
template <bool Reverse = true>
__device__ int32_t next_boundary_face(
    const GeometryParams& geometry,
    const int32_t* edge_ids,
    int32_t begin,
    int32_t end,
    uint64_t bound,
    uint64_t& key) {
  int32_t result = -1;
  key = Reverse ? 0 : UINT64_MAX;
  for (int32_t i = begin; i < end; ++i) {
    TriangleData triangle;
    load_triangle(geometry, edge_ids[i] / 3, triangle);
    const uint64_t candidate = triangle_order_key(triangle, edge_ids[i] / 3);
    const bool in_range = Reverse ? candidate < bound : candidate > bound;
    const bool nearer = Reverse ? candidate > key : candidate < key;
    if (in_range && (result < 0 || nearer)) {
      result = edge_ids[i];
      key = candidate;
    }
  }
  return result;
}

template <typename FragmentEvaluator>
__device__ float boundary_depth_query(
    const GeometryParams& geometry,
    const FragmentEvaluator& evaluator,
    int32_t n,
    int32_t x,
    int32_t y,
    int32_t height,
    int32_t width,
    const int32_t* tile_offsets,
    const int32_t* face_ids,
    const int32_t* edge_ids,
    int32_t begin,
    int32_t end,
    int32_t first_vertex,
    float edge_alpha,
    float edge_depth,
    const TransparentDepthOptions& options) {
  DepthAccumulator plus, minus;
  float plus_transmittance = 1.0f, minus_transmittance = 1.0f;
  const int32_t tile_width = (width + kTileSize - 1) / kTileSize;
  const int32_t tile_height = (height + kTileSize - 1) / kTileSize;
  const int32_t tile = n * tile_height * tile_width + (y / kTileSize) * tile_width + x / kTileSize;
  int32_t actual = tile_offsets[tile];
  const int32_t actual_end = tile_offsets[tile + 1];
  uint64_t query_key;
  int32_t query = next_boundary_face<false>(geometry, edge_ids, begin, end, 0, query_key);
  while (actual < actual_end || query >= 0) {
    TriangleData triangle;
    uint64_t actual_key = UINT64_MAX;
    if (actual < actual_end) {
      load_triangle(geometry, face_ids[actual], triangle);
      actual_key = triangle_order_key(triangle, face_ids[actual]);
    }
    if (query >= 0 && (actual >= actual_end || query_key <= actual_key)) {
      if (actual < actual_end && query_key == actual_key) {
        ++actual;
      }
      load_triangle(geometry, query / 3, triangle);
      if (edge_side(triangle, query % 3, first_vertex) > 0.0f) {
        plus.add(plus_transmittance, edge_alpha, edge_depth, query / 3, options.mode);
        plus_transmittance *= 1.0f - edge_alpha;
      } else {
        minus.add(minus_transmittance, edge_alpha, edge_depth, query / 3, options.mode);
        minus_transmittance *= 1.0f - edge_alpha;
      }
      query = next_boundary_face<false>(geometry, edge_ids, begin, end, query_key, query_key);
      continue;
    }
    const int32_t face = face_ids[actual++];
    const float area = cross2(
        triangle.x[1] - triangle.x[0],
        triangle.y[1] - triangle.y[0],
        triangle.x[2] - triangle.x[0],
        triangle.y[2] - triangle.y[0]);
    float bary[3];
    if (!hard_triangle_coverage(
            triangle, static_cast<float>(x), static_cast<float>(y), area, bary)) {
      continue;
    }
    float inverse_depth = 0.0f;
#pragma unroll
    for (int32_t c = 0; c < 3; ++c) {
      bary[c] /= triangle.z[c];
      inverse_depth += bary[c];
    }
    inverse_depth = clamp_away_from_zero(inverse_depth);
    float material = 0.0f, z = 0.0f;
#pragma unroll
    for (int32_t c = 0; c < 3; ++c) {
      bary[c] /= inverse_depth;
      material += bary[c] * triangle.opacity[c];
      z += bary[c] * triangle.z[c];
    }
    material = fminf(fmaxf(material, 0.0f), 1.0f);
    if (material <= 0.0f) {
      continue;
    }
    float rgb[3], alpha;
    evaluator.evaluate(n, triangle, bary, rgb, alpha);
    alpha = fminf(material * alpha, kAlphaMax);
    plus.add(plus_transmittance, alpha, z, face, options.mode);
    minus.add(minus_transmittance, alpha, z, face, options.mode);
    plus_transmittance *= 1.0f - alpha;
    minus_transmittance *= 1.0f - alpha;
  }
  return plus.value(plus_transmittance, options) - minus.value(minus_transmittance, options);
}

template <typename FragmentEvaluator>
__device__ float boundary_query(
    const GeometryParams& geometry,
    const FragmentEvaluator& evaluator,
    int32_t n,
    int32_t x,
    int32_t y,
    int32_t height,
    int32_t width,
    const int32_t* tile_offsets,
    const int32_t* face_ids,
    const int32_t* edge_ids,
    int32_t begin,
    int32_t end,
    int32_t first_vertex,
    const float* edge_rgb,
    float edge_alpha,
    const float* background,
    const float* grad_color,
    const float* grad_alpha,
    const float* grad_depth,
    const TransparentDepthOptions& depth_options,
    float edge_depth) {
  const int64_t pixel = (static_cast<int64_t>(n) * height + y) * width + x;
  float g[3];
#pragma unroll
  for (int32_t c = 0; c < 3; ++c) {
    g[c] = grad_color[((static_cast<int64_t>(n) * 3 + c) * height + y) * width + x];
  }
  float suffix = g[0] * background[n * 3] + g[1] * background[n * 3 + 1] +
      g[2] * background[n * 3 + 2] - grad_alpha[pixel];
  float jump = 0.0f;
  const float edge_dot = g[0] * edge_rgb[0] + g[1] * edge_rgb[1] + g[2] * edge_rgb[2];
  const int32_t tile_width = (width + kTileSize - 1) / kTileSize;
  const int32_t tile_height = (height + kTileSize - 1) / kTileSize;
  const int32_t tile = n * tile_height * tile_width + (y / kTileSize) * tile_width + x / kTileSize;
  int32_t actual = tile_offsets[tile + 1] - 1;
  const int32_t actual_begin = tile_offsets[tile];
  uint64_t query_key;
  int32_t query = next_boundary_face(geometry, edge_ids, begin, end, UINT64_MAX, query_key);
  while (actual >= actual_begin || query >= 0) {
    TriangleData triangle;
    uint64_t actual_key = 0;
    if (actual >= actual_begin) {
      load_triangle(geometry, face_ids[actual], triangle);
      actual_key = triangle_order_key(triangle, face_ids[actual]);
    }
    if (query >= 0 && (actual < actual_begin || query_key >= actual_key)) {
      if (actual >= actual_begin && query_key == actual_key) {
        --actual;
      }
      load_triangle(geometry, query / 3, triangle);
      if (edge_side(triangle, query % 3, first_vertex) > 0.0f) {
        jump = edge_alpha * (edge_dot - suffix) + (1.0f - edge_alpha) * jump;
      } else {
        jump += edge_alpha * (suffix - edge_dot);
        suffix = edge_alpha * edge_dot + (1.0f - edge_alpha) * suffix;
      }
      query = next_boundary_face(geometry, edge_ids, begin, end, query_key, query_key);
      continue;
    }
    --actual;
    const float area = cross2(
        triangle.x[1] - triangle.x[0],
        triangle.y[1] - triangle.y[0],
        triangle.x[2] - triangle.x[0],
        triangle.y[2] - triangle.y[0]);
    float bary[3];
    if (!hard_triangle_coverage(
            triangle, static_cast<float>(x), static_cast<float>(y), area, bary)) {
      continue;
    }
    float inverse_depth = 0.0f;
#pragma unroll
    for (int32_t c = 0; c < 3; ++c) {
      bary[c] /= triangle.z[c];
      inverse_depth += bary[c];
    }
    inverse_depth = clamp_away_from_zero(inverse_depth);
    float material = 0.0f;
#pragma unroll
    for (int32_t c = 0; c < 3; ++c) {
      bary[c] /= inverse_depth;
      material += bary[c] * triangle.opacity[c];
    }
    material = fminf(fmaxf(material, 0.0f), 1.0f);
    if (material <= 0.0f) {
      continue;
    }
    float rgb[3];
    float alpha;
    evaluator.evaluate(n, triangle, bary, rgb, alpha);
    alpha = fminf(material * alpha, kAlphaMax);
    const float dot = g[0] * rgb[0] + g[1] * rgb[1] + g[2] * rgb[2];
    suffix = alpha * dot + (1.0f - alpha) * suffix;
    jump *= 1.0f - alpha;
  }
  if (grad_depth[pixel] != 0.0f) {
    jump += grad_depth[pixel] *
        boundary_depth_query(
                geometry,
                evaluator,
                n,
                x,
                y,
                height,
                width,
                tile_offsets,
                face_ids,
                edge_ids,
                begin,
                end,
                first_vertex,
                edge_alpha,
                edge_depth,
                depth_options);
  }
  return jump;
}

__global__ void compact_boundary_groups_kernel(
    GeometryParams geometry,
    int32_t edge_count,
    const uint64_t* edge_keys,
    const int32_t* edge_ids,
    int32_t* active_groups,
    int32_t* active_count) {
  const int32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= edge_count || edge_keys[group] == UINT64_MAX ||
      (group > 0 && edge_keys[group - 1] == edge_keys[group])) {
    return;
  }
  int32_t end = group + 1;
  while (end < edge_count && edge_keys[end] == edge_keys[group]) {
    ++end;
  }
  const int32_t edge = edge_ids[group];
  TriangleData triangle;
  load_triangle(geometry, edge / 3, triangle);
  int32_t a = edge % 3;
  int32_t b = (a + 1) % 3;
  if (triangle.vertex_id[a] > triangle.vertex_id[b]) {
    const int32_t tmp = a;
    a = b;
    b = tmp;
  }
  // Shared per-vertex attributes are continuous at a common indexed edge.
  // Opposite projected sides form tessellation, not a coverage boundary.
  if (end - group == 2) {
    TriangleData neighbor;
    load_triangle(geometry, edge_ids[group + 1] / 3, neighbor);
    if ((edge_side(triangle, edge % 3, triangle.vertex_id[a]) > 0.0f) !=
        (edge_side(neighbor, edge_ids[group + 1] % 3, triangle.vertex_id[a]) > 0.0f)) {
      return;
    }
  }
  if (triangle.opacity[a] <= 0.0f && triangle.opacity[b] <= 0.0f) {
    return;
  }
  active_groups[atomicAdd(active_count, 1)] = group;
}

template <typename FragmentEvaluator>
__device__ void boundary_group_backward(
    GeometryParams geometry,
    FragmentEvaluator evaluator,
    int32_t edge_count,
    const uint64_t* edge_keys,
    const int32_t* edge_ids,
    int32_t height,
    int32_t width,
    const int32_t* tile_offsets,
    const int32_t* face_ids,
    const float* background,
    const float* grad_color,
    const float* grad_alpha,
    const float* grad_depth,
    TransparentDepthOptions depth_options,
    float* vertices_grad,
    int32_t group,
    int32_t lane) {
  int32_t end = group + 1;
  while (end < edge_count && edge_keys[end] == edge_keys[group]) {
    ++end;
  }
  const int32_t edge = edge_ids[group];
  const int32_t n = (edge / 3) / geometry.face_count;
  TriangleData triangle;
  load_triangle(geometry, edge / 3, triangle);
  int32_t a = edge % 3;
  int32_t b = (a + 1) % 3;
  if (triangle.vertex_id[a] > triangle.vertex_id[b]) {
    const int32_t tmp = a;
    a = b;
    b = tmp;
  }
  const float dx = triangle.x[b] - triangle.x[a];
  const float dy = triangle.y[b] - triangle.y[a];
  // Sample both grid-line families. Horizontal pairs contribute x motion;
  // vertical pairs contribute y motion, without slope amplification or a
  // second count of the same normal derivative.
#pragma unroll
  for (int32_t axis = 0; axis < 2; ++axis) {
    const float start = axis == 0 ? triangle.y[a] : triangle.x[a];
    const float stop = axis == 0 ? triangle.y[b] : triangle.x[b];
    const float along_start = axis == 0 ? triangle.x[a] : triangle.y[a];
    const float along_delta = axis == 0 ? dx : dy;
    const int32_t scan_size = axis == 0 ? height : width;
    const int32_t pair_size = axis == 0 ? width : height;
    if (start == stop) {
      continue;
    }
    // Clamp in floating point before conversion, including off-screen meshes.
    const int32_t first = static_cast<int32_t>(
        ceilf(fminf(fmaxf(fminf(start, stop), 0.0f), static_cast<float>(scan_size))));
    const int32_t last = static_cast<int32_t>(
        ceilf(fminf(fmaxf(fmaxf(start, stop), 0.0f), static_cast<float>(scan_size))));
    for (int32_t scan = first + lane; scan < last; scan += 32) {
      const float t = (static_cast<float>(scan) - start) / (stop - start);
      const float crossing = along_start + t * along_delta;
      if (!(crossing >= 0.0f && crossing <= static_cast<float>(pair_size - 1))) {
        continue;
      }
      // Classify the nearest pixel with the actual forward edge function.
      // Reconstructing an integer crossing can round to either side, so an
      // equality check on the reconstructed crossing is insufficient.
      const int32_t nearest = static_cast<int32_t>(floorf(crossing + 0.5f));
      const float px = axis == 0 ? static_cast<float>(nearest) : static_cast<float>(scan);
      const float py = axis == 0 ? static_cast<float>(scan) : static_cast<float>(nearest);
      const float side =
          edge_function(triangle.x[a], triangle.y[a], triangle.x[b], triangle.y[b], px, py);
      const bool plus_owns =
          side > 0.0f || (side == 0.0f && (dy < 0.0f || (dy == 0.0f && dx > 0.0f)));
      const bool high_is_plus = axis == 0 ? dy < 0.0f : dx > 0.0f;
      const int32_t low = nearest - (plus_owns == high_is_plus ? 1 : 0);
      if (low < 0 || low + 1 >= pair_size) {
        continue;
      }
      const int32_t x = axis == 0 ? low : scan;
      const int32_t y = axis == 0 ? scan : low;
      float bary[3] = {0.0f, 0.0f, 0.0f};
      bary[a] = (1.0f - t) / triangle.z[a];
      bary[b] = t / triangle.z[b];
      const float inv_depth = clamp_away_from_zero(bary[a] + bary[b]);
      bary[a] /= inv_depth;
      bary[b] /= inv_depth;
      const float material =
          fminf(fmaxf(bary[a] * triangle.opacity[a] + bary[b] * triangle.opacity[b], 0.0f), 1.0f);
      if (material <= 0.0f) {
        continue;
      }
      float rgb[3];
      float alpha;
      evaluator.evaluate(n, triangle, bary, rgb, alpha);
      alpha = fminf(material * alpha, kAlphaMax);
      if (alpha <= 0.0f) {
        continue;
      }
      const float jump0 = boundary_query(
          geometry,
          evaluator,
          n,
          x,
          y,
          height,
          width,
          tile_offsets,
          face_ids,
          edge_ids,
          group,
          end,
          triangle.vertex_id[a],
          rgb,
          alpha,
          background,
          grad_color,
          grad_alpha,
          grad_depth,
          depth_options,
          bary[a] * triangle.z[a] + bary[b] * triangle.z[b]);
      const float jump1 = boundary_query(
          geometry,
          evaluator,
          n,
          x + (axis == 0),
          y + (axis == 1),
          height,
          width,
          tile_offsets,
          face_ids,
          edge_ids,
          group,
          end,
          triangle.vertex_id[a],
          rgb,
          alpha,
          background,
          grad_color,
          grad_alpha,
          grad_depth,
          depth_options,
          bary[a] * triangle.z[a] + bary[b] * triangle.z[b]);
      const float sign = axis == 0 ? (dy > 0.0f ? 1.0f : -1.0f) : (dx > 0.0f ? -1.0f : 1.0f);
      const float force = 0.5f * sign * (jump0 + jump1);
      const int64_t vertex_a =
          static_cast<int64_t>(n) * geometry.vertex_count + triangle.vertex_id[a];
      const int64_t vertex_b =
          static_cast<int64_t>(n) * geometry.vertex_count + triangle.vertex_id[b];
      atomicAdd(vertices_grad + vertex_a * 3 + axis, (1.0f - t) * force);
      atomicAdd(vertices_grad + vertex_b * 3 + axis, t * force);
    }
  }
}

template <typename FragmentEvaluator>
__global__ void boundary_backward_kernel(
    GeometryParams geometry,
    FragmentEvaluator evaluator,
    int32_t edge_count,
    const uint64_t* edge_keys,
    const int32_t* edge_ids,
    const int32_t* active_groups,
    const int32_t* active_count,
    int32_t height,
    int32_t width,
    const int32_t* tile_offsets,
    const int32_t* face_ids,
    const float* background,
    const float* grad_color,
    const float* grad_alpha,
    const float* grad_depth,
    TransparentDepthOptions depth_options,
    float* vertices_grad) {
  const int32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int32_t stride = gridDim.x * blockDim.x / 32;
  for (int32_t index = warp; index < *active_count; index += stride) {
    boundary_group_backward(
        geometry,
        evaluator,
        edge_count,
        edge_keys,
        edge_ids,
        height,
        width,
        tile_offsets,
        face_ids,
        background,
        grad_color,
        grad_alpha,
        grad_depth,
        depth_options,
        vertices_grad,
        active_groups[index],
        threadIdx.x % 32);
  }
}

template <typename FragmentEvaluator>
void launch_boundary_backward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const GeometryParams& geometry,
    const FragmentEvaluator& evaluator,
    int32_t height,
    int32_t width,
    const torch::Tensor& offsets,
    const torch::Tensor& faces,
    const torch::Tensor& background,
    const torch::Tensor& grad_color,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& grad_depth,
    const TransparentDepthOptions& depth_options,
    torch::Tensor& vertices_grad) {
  const int64_t count = vi.numel();
  if (count == 0) {
    return;
  }
  TORCH_CHECK(
      count <= std::numeric_limits<int32_t>::max(),
      "transparent edge gradients: too many face edges");
  TORCH_CHECK(
      v.numel() / 3 <= std::numeric_limits<uint32_t>::max(),
      "transparent edge gradients: too many batched vertices");
  auto keys = torch::empty({count}, vi.options().dtype(torch::kInt64));
  auto ids = torch::empty({count}, vi.options());
  auto sorted_keys = torch::empty_like(keys);
  auto sorted_ids = torch::empty_like(ids);
  const auto stream = at::cuda::getCurrentCUDAStream();
  boundary_edge_keys_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      geometry,
      static_cast<int32_t>(count),
      reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
      ids.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  size_t bytes = 0;
  C10_CUDA_CHECK(
      cub::DeviceRadixSort::SortPairs(
          nullptr,
          bytes,
          reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
          reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
          ids.data_ptr<int32_t>(),
          sorted_ids.data_ptr<int32_t>(),
          count,
          0,
          64,
          stream));
  auto temporary = torch::empty({static_cast<int64_t>(bytes)}, vi.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(
      cub::DeviceRadixSort::SortPairs(
          temporary.data_ptr<uint8_t>(),
          bytes,
          reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
          reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
          ids.data_ptr<int32_t>(),
          sorted_ids.data_ptr<int32_t>(),
          count,
          0,
          64,
          stream));
  auto active_groups = torch::empty_like(ids);
  auto active_count = torch::zeros({1}, vi.options());
  compact_boundary_groups_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      geometry,
      static_cast<int32_t>(count),
      reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
      sorted_ids.data_ptr<int32_t>(),
      active_groups.data_ptr<int32_t>(),
      active_count.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int32_t blocks = std::min<int64_t>(
      (count + 3) / 4, 4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  boundary_backward_kernel<<<blocks, 128, 0, stream>>>(
      geometry,
      evaluator,
      static_cast<int32_t>(count),
      reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
      sorted_ids.data_ptr<int32_t>(),
      active_groups.data_ptr<int32_t>(),
      active_count.data_ptr<int32_t>(),
      height,
      width,
      offsets.data_ptr<int32_t>(),
      faces.data_ptr<int32_t>(),
      background.data_ptr<float>(),
      grad_color.data_ptr<float>(),
      grad_alpha.data_ptr<float>(),
      grad_depth.data_ptr<float>(),
      depth_options,
      vertices_grad.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename FragmentEvaluator>
__global__ void render_tiles_kernel(
    GeometryParams geometry,
    FragmentEvaluator fragment_evaluator,
    int32_t image_height,
    int32_t image_width,
    int32_t tile_height,
    int32_t tile_width,
    float transmittance_epsilon,
    const int32_t* tile_offsets,
    const int32_t* sorted_face_ids,
    const float* background,
    float* output_color,
    float* output_alpha,
    float* output_depth,
    int32_t* saved_median_face,
    TransparentDepthOptions depth_options,
    float* saved_transmittance,
    int32_t* saved_transmittance_exponent) {
  const int32_t n = blockIdx.z;
  const int32_t tile_x = blockIdx.x;
  const int32_t tile_y = blockIdx.y;
  const int32_t x = tile_x * kTileSize + threadIdx.x;
  const int32_t y = tile_y * kTileSize + threadIdx.y;
  const int32_t thread_index = threadIdx.y * kTileSize + threadIdx.x;
  const int32_t tile_id = n * tile_height * tile_width + tile_y * tile_width + tile_x;
  const int32_t range_start = tile_offsets[tile_id];
  const int32_t range_end = tile_offsets[tile_id + 1];

  __shared__ TriangleData triangles[kFacesPerBatch];
  const bool inside = x < image_width && y < image_height;
  bool done = !inside;
  float transmittance = 1.0f;
  ScaledTransmittance backward_transmittance;
  float color[3] = {0.0f, 0.0f, 0.0f};
  DepthAccumulator depth;

  const int32_t candidate_count = range_end - range_start;
  const int32_t batch_count = (candidate_count + kFacesPerBatch - 1) / kFacesPerBatch;
  for (int32_t batch = 0; batch < batch_count; ++batch) {
    const int32_t done_count = __syncthreads_count(done);
    if (done_count == kTileThreads) {
      break;
    }
    const int32_t batch_start = range_start + batch * kFacesPerBatch;
    const int32_t faces_in_batch = min(kFacesPerBatch, range_end - batch_start);
    if (thread_index < faces_in_batch) {
      load_triangle(geometry, sorted_face_ids[batch_start + thread_index], triangles[thread_index]);
    }
    __syncthreads();

    if (done) {
      continue;
    }
    const float px = static_cast<float>(x);
    const float py = static_cast<float>(y);
    for (int32_t face_in_batch = 0; face_in_batch < faces_in_batch; ++face_in_batch) {
      const TriangleData& triangle = triangles[face_in_batch];
      const float edge01_x = triangle.x[1] - triangle.x[0];
      const float edge01_y = triangle.y[1] - triangle.y[0];
      const float edge02_x = triangle.x[2] - triangle.x[0];
      const float edge02_y = triangle.y[2] - triangle.y[0];
      const float area = cross2(edge01_x, edge01_y, edge02_x, edge02_y);
      if (fabsf(area) <= 1.0e-12f) {
        continue;
      }
      float screen_bary[3];
      if (!hard_triangle_coverage(triangle, px, py, area, screen_bary)) {
        continue;
      }

      const float weighted_bary0 = screen_bary[0] / clamp_away_from_zero(triangle.z[0]);
      const float weighted_bary1 = screen_bary[1] / clamp_away_from_zero(triangle.z[1]);
      const float weighted_bary2 = screen_bary[2] / clamp_away_from_zero(triangle.z[2]);
      const float inverse_depth =
          clamp_away_from_zero(weighted_bary0 + weighted_bary1 + weighted_bary2);
      const float bary[3] = {
          weighted_bary0 / inverse_depth,
          weighted_bary1 / inverse_depth,
          weighted_bary2 / inverse_depth};

      float fragment_rgb[3];
      float appearance_alpha = 1.0f;
      fragment_evaluator.evaluate(n, triangle, bary, fragment_rgb, appearance_alpha);
      float material_opacity = 0.0f;
#pragma unroll
      for (int32_t corner = 0; corner < 3; ++corner) {
        material_opacity += bary[corner] * triangle.opacity[corner];
      }
      material_opacity = fminf(fmaxf(material_opacity, 0.0f), 1.0f);
      const float alpha = fminf(material_opacity * appearance_alpha, kAlphaMax);
      if (alpha <= 0.0f) {
        continue;
      }
      const float visibility = transmittance * alpha;
#pragma unroll
      for (int32_t channel = 0; channel < 3; ++channel) {
        color[channel] += visibility * fragment_rgb[channel];
      }
      const float z = bary[0] * triangle.z[0] + bary[1] * triangle.z[1] + bary[2] * triangle.z[2];
      depth.add(
          transmittance,
          alpha,
          z,
          sorted_face_ids[batch_start + face_in_batch],
          depth_options.mode);
      transmittance *= 1.0f - alpha;
      if (saved_transmittance != nullptr) {
        backward_transmittance.multiply(1.0f - alpha);
      }
      if (transmittance_epsilon > 0.0f && transmittance <= transmittance_epsilon) {
        done = true;
        break;
      }
    }
  }

  if (inside) {
    const int64_t pixel = static_cast<int64_t>(n) * image_height * image_width +
        static_cast<int64_t>(y) * image_width + x;
#pragma unroll
    for (int32_t channel = 0; channel < 3; ++channel) {
      output_color
          [(static_cast<int64_t>(n) * 3 + channel) * image_height * image_width +
           static_cast<int64_t>(y) * image_width + x] =
              color[channel] + transmittance * background[n * 3 + channel];
    }
    output_alpha[pixel] = depth.alpha;
    output_depth[pixel] = depth.value(transmittance, depth_options);
    if (saved_median_face != nullptr) {
      saved_median_face[pixel] = depth.median_face;
    }
    if (saved_transmittance != nullptr) {
      saved_transmittance[pixel] = backward_transmittance.mantissa;
      saved_transmittance_exponent[pixel] = backward_transmittance.exponent;
    }
  }
}

template <typename FragmentEvaluator>
__global__ void render_tiles_backward_kernel(
    GeometryParams geometry,
    FragmentEvaluator fragment_evaluator,
    int32_t image_height,
    int32_t image_width,
    int32_t tile_height,
    int32_t tile_width,
    const int32_t* tile_offsets,
    const int32_t* sorted_face_ids,
    const float* background,
    const float* saved_transmittance,
    const int32_t* saved_transmittance_exponent,
    const float* grad_color,
    const float* grad_alpha,
    const float* grad_depth,
    const float* saved_alpha,
    const float* saved_depth,
    const int32_t* saved_median_face,
    TransparentDepthOptions depth_options,
    float* vertices_grad,
    float* uv_grad,
    float* reference_uv_grad,
    float* opacity_grad,
    float* background_grad) {
  const int32_t n = blockIdx.z;
  const int32_t tile_x = blockIdx.x;
  const int32_t tile_y = blockIdx.y;
  const int32_t x = tile_x * kTileSize + threadIdx.x;
  const int32_t y = tile_y * kTileSize + threadIdx.y;
  const int32_t thread_index = threadIdx.y * kTileSize + threadIdx.x;
  const int32_t tile_id = n * tile_height * tile_width + tile_y * tile_width + tile_x;
  const int32_t range_start = tile_offsets[tile_id];
  const int32_t range_end = tile_offsets[tile_id + 1];

  __shared__ TriangleData triangles[kFacesPerBatch];
  __shared__ float shared_mlp_weight_grad[kMaxMlpWeightParams];
  __shared__ float shared_mlp_bias_grad[kMaxMlpBiasParams];
  FragmentEvaluator block_evaluator = fragment_evaluator;
  const int32_t shared_weight_count = block_evaluator.shared_weight_grad_count();
  const int32_t shared_bias_count = block_evaluator.shared_bias_grad_count();
  for (int32_t index = thread_index; index < shared_weight_count; index += kTileThreads) {
    shared_mlp_weight_grad[index] = 0.0f;
  }
  for (int32_t index = thread_index; index < shared_bias_count; index += kTileThreads) {
    shared_mlp_bias_grad[index] = 0.0f;
  }
  __syncthreads();
  block_evaluator.bind_shared_mlp_grads(shared_mlp_weight_grad, shared_mlp_bias_grad);
  const bool inside = x < image_width && y < image_height;
  const int64_t pixel = static_cast<int64_t>(n) * image_height * image_width +
      static_cast<int64_t>(min(y, image_height - 1)) * image_width + min(x, image_width - 1);
  float color_grad[3] = {0.0f, 0.0f, 0.0f};
  ScaledTransmittance transmittance_after;
  float transmittance_grad_after = 0.0f;
  float depth_grad = 0.0f;
  float weighted_depth_grad = 0.0f;
  int32_t median_face = -1;
  if (inside) {
#pragma unroll
    for (int32_t channel = 0; channel < 3; ++channel) {
      color_grad[channel] = grad_color
          [(static_cast<int64_t>(n) * 3 + channel) * image_height * image_width +
           static_cast<int64_t>(y) * image_width + x];
    }
    depth_grad = grad_depth[pixel];
    float alpha_grad = grad_alpha[pixel];
    if (depth_options.mode == TransparentDepthMode::ForegroundMean) {
      const float alpha = saved_alpha[pixel];
      if (alpha > depth_options.alpha_threshold) {
        weighted_depth_grad = depth_grad / alpha;
        alpha_grad -= weighted_depth_grad * saved_depth[pixel];
      }
    } else if (depth_options.mode == TransparentDepthMode::Median) {
      median_face = saved_median_face[pixel];
    } else {
      weighted_depth_grad = depth_grad;
    }
    transmittance_after.mantissa = saved_transmittance[pixel];
    transmittance_after.exponent = saved_transmittance_exponent[pixel];
    transmittance_grad_after = color_grad[0] * background[n * 3 + 0] +
        color_grad[1] * background[n * 3 + 1] + color_grad[2] * background[n * 3 + 2] - alpha_grad;
    if (depth_options.mode == TransparentDepthMode::Expected) {
      transmittance_grad_after += depth_grad * depth_options.background;
    }
#pragma unroll
    for (int32_t channel = 0; channel < 3; ++channel) {
      atomicAdd(
          background_grad + n * 3 + channel, color_grad[channel] * transmittance_after.value());
    }
  }

  const int32_t candidate_count = range_end - range_start;
  const int32_t batch_count = (candidate_count + kFacesPerBatch - 1) / kFacesPerBatch;
  for (int32_t batch = 0; batch < batch_count; ++batch) {
    const int32_t batch_end = range_end - batch * kFacesPerBatch;
    const int32_t batch_start = max(range_start, batch_end - kFacesPerBatch);
    const int32_t faces_in_batch = batch_end - batch_start;
    if (thread_index < faces_in_batch) {
      load_triangle(geometry, sorted_face_ids[batch_start + thread_index], triangles[thread_index]);
    }
    __syncthreads();

    if (inside) {
      const float px = static_cast<float>(x);
      const float py = static_cast<float>(y);
      for (int32_t face_in_batch = faces_in_batch - 1; face_in_batch >= 0; --face_in_batch) {
        const TriangleData& triangle = triangles[face_in_batch];
        const float edge01_x = triangle.x[1] - triangle.x[0];
        const float edge01_y = triangle.y[1] - triangle.y[0];
        const float edge02_x = triangle.x[2] - triangle.x[0];
        const float edge02_y = triangle.y[2] - triangle.y[0];
        const float area = cross2(edge01_x, edge01_y, edge02_x, edge02_y);
        if (fabsf(area) <= 1.0e-12f) {
          continue;
        }
        float screen_bary[3];
        if (!hard_triangle_coverage(triangle, px, py, area, screen_bary)) {
          continue;
        }

        const float weighted_bary[3] = {
            screen_bary[0] / clamp_away_from_zero(triangle.z[0]),
            screen_bary[1] / clamp_away_from_zero(triangle.z[1]),
            screen_bary[2] / clamp_away_from_zero(triangle.z[2])};
        const float inverse_depth =
            clamp_away_from_zero(weighted_bary[0] + weighted_bary[1] + weighted_bary[2]);
        const float bary[3] = {
            weighted_bary[0] / inverse_depth,
            weighted_bary[1] / inverse_depth,
            weighted_bary[2] / inverse_depth};

        float fragment_rgb[3];
        float appearance_alpha = 1.0f;
        block_evaluator.evaluate(n, triangle, bary, fragment_rgb, appearance_alpha);
        float raw_material_opacity = 0.0f;
#pragma unroll
        for (int32_t corner = 0; corner < 3; ++corner) {
          raw_material_opacity += bary[corner] * triangle.opacity[corner];
        }
        const float material_opacity = fminf(fmaxf(raw_material_opacity, 0.0f), 1.0f);
        const float raw_alpha = material_opacity * appearance_alpha;
        const float alpha = fminf(raw_alpha, kAlphaMax);
        if (alpha <= 0.0f) {
          continue;
        }

        const float transmittance_before = transmittance_after.divide(1.0f - alpha);
        const float fragment_depth =
            bary[0] * triangle.z[0] + bary[1] * triangle.z[1] + bary[2] * triangle.z[2];
        const float fragment_depth_grad = depth_options.mode == TransparentDepthMode::Median
            ? (sorted_face_ids[batch_start + face_in_batch] == median_face ? depth_grad : 0.0f)
            : transmittance_before * alpha * weighted_depth_grad;
        const float color_dot_grad = weighted_depth_grad * fragment_depth +
            fragment_rgb[0] * color_grad[0] + fragment_rgb[1] * color_grad[1] +
            fragment_rgb[2] * color_grad[2];
        const float alpha_grad = transmittance_before * (color_dot_grad - transmittance_grad_after);
        const float fragment_rgb_grad[3] = {
            transmittance_before * alpha * color_grad[0],
            transmittance_before * alpha * color_grad[1],
            transmittance_before * alpha * color_grad[2]};
        const float transmittance_grad_before =
            alpha * color_dot_grad + (1.0f - alpha) * transmittance_grad_after;

        const bool alpha_is_differentiable = raw_alpha > 0.0f && raw_alpha <= kAlphaMax;
        const float raw_alpha_grad = alpha_is_differentiable ? alpha_grad : 0.0f;
        const float material_opacity_grad =
            raw_material_opacity >= 0.0f && raw_material_opacity <= 1.0f
            ? raw_alpha_grad * appearance_alpha
            : 0.0f;
        const float appearance_alpha_grad = raw_alpha_grad * material_opacity;

        FragmentAttributeGrads attribute_grad{};
        block_evaluator.backward(
            n,
            triangle,
            bary,
            fragment_rgb_grad,
            appearance_alpha_grad,
            appearance_alpha,
            attribute_grad);
#pragma unroll
        for (int32_t corner = 0; corner < 3; ++corner) {
          attribute_grad.bary[corner] += material_opacity_grad * triangle.opacity[corner] +
              fragment_depth_grad * triangle.z[corner];
          const int64_t vertex =
              static_cast<int64_t>(n) * geometry.vertex_count + triangle.vertex_id[corner];
          atomicAdd(opacity_grad + vertex, material_opacity_grad * bary[corner]);
          atomicAdd(uv_grad + vertex * 2 + 0, attribute_grad.u[corner]);
          atomicAdd(uv_grad + vertex * 2 + 1, attribute_grad.v[corner]);
          if (reference_uv_grad != nullptr) {
            atomicAdd(reference_uv_grad + vertex * 2 + 0, attribute_grad.reference_u[corner]);
            atomicAdd(reference_uv_grad + vertex * 2 + 1, attribute_grad.reference_v[corner]);
          }
        }

        float x_grad[3] = {0.0f, 0.0f, 0.0f};
        float y_grad[3] = {0.0f, 0.0f, 0.0f};
        float z_grad[3] = {
            fragment_depth_grad * bary[0],
            fragment_depth_grad * bary[1],
            fragment_depth_grad * bary[2]};
        perspective_barycentric_backward(
            triangle,
            px,
            py,
            screen_bary,
            bary,
            inverse_depth,
            attribute_grad.bary,
            x_grad,
            y_grad,
            z_grad);
#pragma unroll
        for (int32_t corner = 0; corner < 3; ++corner) {
          const int64_t vertex =
              static_cast<int64_t>(n) * geometry.vertex_count + triangle.vertex_id[corner];
          atomicAdd(vertices_grad + vertex * 3 + 0, x_grad[corner]);
          atomicAdd(vertices_grad + vertex * 3 + 1, y_grad[corner]);
          atomicAdd(vertices_grad + vertex * 3 + 2, z_grad[corner]);
        }

        transmittance_grad_after = transmittance_grad_before;
      }
    }
    __syncthreads();
  }
  __syncthreads();
  block_evaluator.flush_shared_mlp_grads(thread_index, kTileThreads);
}

void check_cuda_float_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), "transparent render: ", name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), "transparent render: ", name, " must be a CUDA tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32,
      "transparent render: ",
      name,
      " must have float32 dtype, but got ",
      tensor.scalar_type());
  TORCH_CHECK(tensor.is_contiguous(), "transparent render: ", name, " must be contiguous");
}

void check_common_inputs(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation) {
  check_cuda_float_contiguous(v, "v");
  check_cuda_float_contiguous(uv, "uv");
  check_cuda_float_contiguous(opacity, "opacity");
  check_cuda_float_contiguous(background, "background");
  TORCH_CHECK(vi.defined() && vi.is_cuda(), "transparent render: vi must be a CUDA tensor");
  TORCH_CHECK(
      vi.scalar_type() == torch::kInt32,
      "transparent render: vi must have int32 dtype, but got ",
      vi.scalar_type());
  TORCH_CHECK(vi.is_contiguous(), "transparent render: vi must be contiguous");
  TORCH_CHECK(v.dim() == 3 && v.size(2) == 3, "transparent render: expected v [N, V, 3]");
  TORCH_CHECK(
      vi.dim() == 3 && vi.size(0) == v.size(0) && vi.size(2) == 3,
      "transparent render: expected vi [N, F, 3]");
  TORCH_CHECK(
      uv.dim() == 3 && uv.size(0) == v.size(0) && uv.size(1) == v.size(1) && uv.size(2) == 2,
      "transparent render: expected uv [N, V, 2]");
  TORCH_CHECK(
      opacity.dim() == 2 && opacity.size(0) == v.size(0) && opacity.size(1) == v.size(1),
      "transparent render: expected opacity [N, V]");
  TORCH_CHECK(
      background.dim() == 2 && background.size(0) == v.size(0) && background.size(1) == 3,
      "transparent render: expected background [N, 3]");
  TORCH_CHECK(height > 0 && width > 0, "transparent render: output dimensions must be positive");
  TORCH_CHECK(
      interpolation_mode >= static_cast<int64_t>(TextureInterpolation::Bilinear) &&
          interpolation_mode <= static_cast<int64_t>(TextureInterpolation::Bicubic),
      "transparent render: invalid interpolation mode");
  TORCH_CHECK(
      padding_mode >= static_cast<int64_t>(GridSamplerPadding::Zeros) &&
          padding_mode <= static_cast<int64_t>(GridSamplerPadding::Reflection),
      "transparent render: invalid padding mode");
  TORCH_CHECK(
      transmittance_epsilon >= 0.0 && transmittance_epsilon < 1.0,
      "transparent render: transmittance_epsilon must be in [0, 1)");
  TORCH_CHECK(
      rgb_activation >= 0 && rgb_activation < kActivationModeCount,
      "transparent render: invalid RGB activation");
  TORCH_CHECK(
      alpha_activation >= 0 && alpha_activation < kActivationModeCount,
      "transparent render: invalid alpha activation");
  TORCH_CHECK(
      v.size(0) <= std::numeric_limits<int32_t>::max() &&
          v.size(1) <= std::numeric_limits<int32_t>::max() &&
          vi.size(1) <= std::numeric_limits<int32_t>::max() &&
          height <= std::numeric_limits<int32_t>::max() &&
          width <= std::numeric_limits<int32_t>::max(),
      "transparent render: tensor dimensions exceed int32 indexing limits");
  TORCH_CHECK(
      vi.device() == v.device() && uv.device() == v.device() && opacity.device() == v.device() &&
          background.device() == v.device(),
      "transparent render: all inputs must share a CUDA device");
}

TileBins build_tile_bins(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    int32_t image_height,
    int32_t image_width) {
  const int32_t batch_size = static_cast<int32_t>(v.size(0));
  const int32_t vertex_count = static_cast<int32_t>(v.size(1));
  const int32_t face_count = static_cast<int32_t>(vi.size(1));
  const int32_t tile_height = (image_height + kTileSize - 1) / kTileSize;
  const int32_t tile_width = (image_width + kTileSize - 1) / kTileSize;
  const int64_t tiles_per_image = static_cast<int64_t>(tile_height) * tile_width;
  const int64_t total_tiles = static_cast<int64_t>(batch_size) * tiles_per_image;
  TORCH_CHECK(
      total_tiles < static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "transparent render: too many batch/tile combinations");

  const int64_t face_element_count = static_cast<int64_t>(batch_size) * face_count;
  TORCH_CHECK(
      face_element_count <= std::numeric_limits<int32_t>::max(),
      "transparent render: flattened face count exceeds int32 indexing");
  auto int_options = vi.options().dtype(torch::kInt32);
  auto long_options = vi.options().dtype(torch::kInt64);
  auto tile_bounds = torch::empty({face_element_count, 4}, int_options);
  auto tile_counts = torch::empty({face_element_count}, int_options);
  auto face_depths = torch::empty({face_element_count}, v.options());

  if (face_element_count > 0) {
    const int32_t blocks =
        static_cast<int32_t>((face_element_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    compute_tile_bounds_kernel<<<blocks, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        face_element_count,
        v.data_ptr<float>(),
        vi.data_ptr<int32_t>(),
        vertex_count,
        face_count,
        image_height,
        image_width,
        tile_height,
        tile_width,
        tile_bounds.data_ptr<int32_t>(),
        tile_counts.data_ptr<int32_t>(),
        face_depths.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  torch::Tensor cumulative_tile_counts;
  int64_t intersection_count = 0;
  if (face_element_count > 0) {
    cumulative_tile_counts = at::cumsum(tile_counts, 0, at::kLong);
    intersection_count = cumulative_tile_counts[-1].item<int64_t>();
  } else {
    cumulative_tile_counts = torch::empty({0}, long_options);
  }
  TORCH_CHECK(
      intersection_count <= std::numeric_limits<int32_t>::max(),
      "transparent render: tile/triangle intersection list exceeds int32 indexing");

  auto keys = torch::empty({intersection_count}, long_options);
  auto face_ids = torch::empty({intersection_count}, int_options);
  if (intersection_count > 0) {
    const int32_t blocks =
        static_cast<int32_t>((face_element_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    emit_tile_keys_kernel<<<blocks, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        face_element_count,
        face_count,
        tile_width,
        static_cast<int32_t>(tiles_per_image),
        tile_bounds.data_ptr<int32_t>(),
        cumulative_tile_counts.data_ptr<int64_t>(),
        face_depths.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
        face_ids.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto sorted_keys = torch::empty_like(keys);
  auto sorted_face_ids = torch::empty_like(face_ids);
  if (intersection_count > 0) {
    size_t temporary_bytes = 0;
    C10_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            temporary_bytes,
            reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
            reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
            face_ids.data_ptr<int32_t>(),
            sorted_face_ids.data_ptr<int32_t>(),
            intersection_count,
            0,
            64,
            at::cuda::getCurrentCUDAStream()));
    auto temporary =
        torch::empty({static_cast<int64_t>(temporary_bytes)}, vi.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            temporary.data_ptr<uint8_t>(),
            temporary_bytes,
            reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
            reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
            face_ids.data_ptr<int32_t>(),
            sorted_face_ids.data_ptr<int32_t>(),
            intersection_count,
            0,
            64,
            at::cuda::getCurrentCUDAStream()));
  }

  auto offsets = torch::empty({total_tiles + 1}, int_options);
  const int32_t offset_count = static_cast<int32_t>(total_tiles + 1);
  const int32_t offset_blocks = (offset_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  find_tile_offsets_kernel<<<
      offset_blocks,
      kThreadsPerBlock,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      static_cast<int32_t>(total_tiles),
      intersection_count,
      reinterpret_cast<const uint64_t*>(sorted_keys.data_ptr<int64_t>()),
      offsets.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {offsets, sorted_face_ids};
}

template <typename FragmentEvaluator>
std::vector<torch::Tensor> launch_render(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor* reference_uv,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int32_t image_height,
    int32_t image_width,
    float transmittance_epsilon,
    const FragmentEvaluator& fragment_evaluator,
    const TransparentDepthOptions& depth_options,
    bool save_transmittance) {
  const c10::cuda::CUDAGuard device_guard(v.device());
  const int32_t batch_size = static_cast<int32_t>(v.size(0));
  const int32_t vertex_count = static_cast<int32_t>(v.size(1));
  const int32_t face_count = static_cast<int32_t>(vi.size(1));
  const int32_t tile_height = (image_height + kTileSize - 1) / kTileSize;
  const int32_t tile_width = (image_width + kTileSize - 1) / kTileSize;
  TileBins bins = build_tile_bins(v, vi, image_height, image_width);

  auto color = torch::empty({batch_size, 3, image_height, image_width}, v.options());
  auto alpha = torch::empty({batch_size, 1, image_height, image_width}, v.options());
  auto depth = torch::empty_like(alpha);
  auto median_face = save_transmittance && depth_options.mode == TransparentDepthMode::Median
      ? torch::empty_like(alpha, alpha.options().dtype(torch::kInt32))
      : torch::Tensor();
  auto saved_transmittance = save_transmittance ? torch::empty_like(alpha) : torch::Tensor();
  auto saved_transmittance_exponent = save_transmittance
      ? torch::empty_like(alpha, alpha.options().dtype(torch::kInt32))
      : torch::Tensor();
  if (batch_size > 0) {
    GeometryParams geometry{
        v.data_ptr<float>(),
        vi.data_ptr<int32_t>(),
        uv.data_ptr<float>(),
        reference_uv == nullptr ? nullptr : reference_uv->data_ptr<float>(),
        opacity.data_ptr<float>(),
        vertex_count,
        face_count};
    const dim3 block(kTileSize, kTileSize, 1);
    const dim3 grid(tile_width, tile_height, batch_size);
    render_tiles_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        geometry,
        fragment_evaluator,
        image_height,
        image_width,
        tile_height,
        tile_width,
        transmittance_epsilon,
        bins.offsets.data_ptr<int32_t>(),
        bins.face_ids.data_ptr<int32_t>(),
        background.data_ptr<float>(),
        color.data_ptr<float>(),
        alpha.data_ptr<float>(),
        depth.data_ptr<float>(),
        median_face.defined() ? median_face.data_ptr<int32_t>() : nullptr,
        depth_options,
        save_transmittance ? saved_transmittance.data_ptr<float>() : nullptr,
        save_transmittance ? saved_transmittance_exponent.data_ptr<int32_t>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {
      color,
      alpha,
      depth,
      bins.offsets,
      bins.face_ids,
      saved_transmittance,
      saved_transmittance_exponent,
      median_face};
}

void check_texture(
    const torch::Tensor& texture,
    int64_t batch_size,
    const char* name,
    bool allow_zero_channels = false) {
  check_cuda_float_contiguous(texture, name);
  TORCH_CHECK(
      texture.dim() == 4 && (texture.size(0) == 1 || texture.size(0) == batch_size) &&
          (texture.size(1) > 0 || (allow_zero_channels && texture.size(1) == 0)) &&
          texture.size(2) > 0 && texture.size(3) > 0,
      "transparent render: ",
      name,
      " must have shape [N or 1, C, H, W] with non-empty spatial dimensions");
}

void check_backward_inputs(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& sorted_face_ids,
    const torch::Tensor& saved_transmittance,
    const torch::Tensor& saved_transmittance_exponent,
    const torch::Tensor& grad_color,
    const torch::Tensor& grad_alpha,
    int64_t interpolation_mode,
    int64_t padding_mode,
    int64_t rgb_activation,
    int64_t alpha_activation) {
  TORCH_CHECK(
      saved_transmittance.dim() == 4 && saved_transmittance.size(0) == v.size(0) &&
          saved_transmittance.size(1) == 1,
      "transparent render backward: expected saved_transmittance [N, 1, H, W]");
  const int64_t height = saved_transmittance.size(2);
  const int64_t width = saved_transmittance.size(3);
  check_common_inputs(
      v,
      vi,
      uv,
      opacity,
      background,
      height,
      width,
      interpolation_mode,
      padding_mode,
      0.0,
      rgb_activation,
      alpha_activation);
  check_cuda_float_contiguous(saved_transmittance, "saved_transmittance");
  TORCH_CHECK(
      saved_transmittance_exponent.is_cuda() &&
          saved_transmittance_exponent.scalar_type() == torch::kInt32 &&
          saved_transmittance_exponent.is_contiguous() &&
          saved_transmittance_exponent.sizes() == saved_transmittance.sizes() &&
          saved_transmittance_exponent.device() == v.device(),
      "transparent render backward: invalid saved transmittance exponent");
  check_cuda_float_contiguous(grad_color, "grad_color");
  check_cuda_float_contiguous(grad_alpha, "grad_alpha");
  TORCH_CHECK(
      grad_color.dim() == 4 && grad_color.size(0) == v.size(0) && grad_color.size(1) == 3 &&
          grad_color.size(2) == height && grad_color.size(3) == width,
      "transparent render backward: invalid grad_color shape");
  TORCH_CHECK(
      grad_alpha.sizes() == saved_transmittance.sizes(),
      "transparent render backward: invalid grad_alpha shape");
  TORCH_CHECK(
      tile_offsets.defined() && tile_offsets.is_cuda() &&
          tile_offsets.scalar_type() == torch::kInt32 && tile_offsets.is_contiguous(),
      "transparent render backward: tile_offsets must be contiguous CUDA int32");
  TORCH_CHECK(
      sorted_face_ids.defined() && sorted_face_ids.is_cuda() &&
          sorted_face_ids.scalar_type() == torch::kInt32 && sorted_face_ids.is_contiguous(),
      "transparent render backward: sorted_face_ids must be contiguous CUDA int32");
  TORCH_CHECK(
      saved_transmittance.device() == v.device() && grad_color.device() == v.device() &&
          grad_alpha.device() == v.device() && tile_offsets.device() == v.device() &&
          sorted_face_ids.device() == v.device(),
      "transparent render backward: all tensors must share a CUDA device");
}

template <typename FragmentEvaluator>
std::vector<torch::Tensor> launch_render_backward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor* reference_uv,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& sorted_face_ids,
    const torch::Tensor& saved_transmittance,
    const torch::Tensor& saved_transmittance_exponent,
    const torch::Tensor& grad_color,
    const torch::Tensor& grad_alpha,
    const FragmentEvaluator& fragment_evaluator,
    const torch::Tensor& saved_alpha,
    const torch::Tensor& saved_depth,
    const torch::Tensor& saved_median_face,
    const torch::Tensor& grad_depth,
    const TransparentDepthOptions& depth_options,
    bool edge_gradients) {
  const c10::cuda::CUDAGuard device_guard(v.device());
  const int32_t batch_size = static_cast<int32_t>(v.size(0));
  const int32_t vertex_count = static_cast<int32_t>(v.size(1));
  const int32_t face_count = static_cast<int32_t>(vi.size(1));
  const int32_t image_height = static_cast<int32_t>(saved_transmittance.size(2));
  const int32_t image_width = static_cast<int32_t>(saved_transmittance.size(3));
  const int32_t tile_height = (image_height + kTileSize - 1) / kTileSize;
  const int32_t tile_width = (image_width + kTileSize - 1) / kTileSize;
  const int64_t expected_offsets = static_cast<int64_t>(batch_size) * tile_height * tile_width + 1;
  TORCH_CHECK(
      tile_offsets.numel() == expected_offsets,
      "transparent render backward: tile offset shape does not match image size");

  auto vertices_grad = torch::zeros_like(v);
  auto uv_grad = torch::zeros_like(uv);
  torch::Tensor reference_uv_grad =
      reference_uv == nullptr ? torch::Tensor() : torch::zeros_like(*reference_uv);
  auto opacity_grad = torch::zeros_like(opacity);
  auto background_grad = torch::zeros_like(background);
  if (batch_size > 0) {
    GeometryParams geometry{
        v.data_ptr<float>(),
        vi.data_ptr<int32_t>(),
        uv.data_ptr<float>(),
        reference_uv == nullptr ? nullptr : reference_uv->data_ptr<float>(),
        opacity.data_ptr<float>(),
        vertex_count,
        face_count};
    const dim3 block(kTileSize, kTileSize, 1);
    const dim3 grid(tile_width, tile_height, batch_size);
    render_tiles_backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        geometry,
        fragment_evaluator,
        image_height,
        image_width,
        tile_height,
        tile_width,
        tile_offsets.data_ptr<int32_t>(),
        sorted_face_ids.data_ptr<int32_t>(),
        background.data_ptr<float>(),
        saved_transmittance.data_ptr<float>(),
        saved_transmittance_exponent.data_ptr<int32_t>(),
        grad_color.data_ptr<float>(),
        grad_alpha.data_ptr<float>(),
        grad_depth.data_ptr<float>(),
        saved_alpha.data_ptr<float>(),
        saved_depth.data_ptr<float>(),
        saved_median_face.defined() ? saved_median_face.data_ptr<int32_t>() : nullptr,
        depth_options,
        vertices_grad.data_ptr<float>(),
        uv_grad.data_ptr<float>(),
        reference_uv == nullptr ? nullptr : reference_uv_grad.data_ptr<float>(),
        opacity_grad.data_ptr<float>(),
        background_grad.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (edge_gradients) {
      launch_boundary_backward(
          v,
          vi,
          geometry,
          fragment_evaluator.forward,
          image_height,
          image_width,
          tile_offsets,
          sorted_face_ids,
          background,
          grad_color,
          grad_alpha,
          grad_depth,
          depth_options,
          vertices_grad);
    }
  }
  return {vertices_grad, uv_grad, reference_uv_grad, opacity_grad, background_grad};
}

} // namespace

std::vector<torch::Tensor> transparent_render_with_texture_cuda_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& texture,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    const TransparentDepthOptions& depth_options,
    bool save_transmittance) {
  check_common_inputs(
      v,
      vi,
      uv,
      opacity,
      background,
      height,
      width,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation);
  check_texture(texture, v.size(0), "texture");
  TORCH_CHECK(
      texture.size(1) == 3 || texture.size(1) == 4,
      "transparent render: render_with_texture requires RGB or RGBA input");
  TORCH_CHECK(texture.device() == v.device(), "transparent render: all inputs must share a device");
  TextureFragmentEvaluator fragment_evaluator{
      texture.data_ptr<float>(),
      static_cast<int32_t>(texture.size(0)),
      static_cast<int32_t>(texture.size(1)),
      static_cast<int32_t>(texture.size(2)),
      static_cast<int32_t>(texture.size(3)),
      static_cast<int32_t>(interpolation_mode),
      static_cast<int32_t>(padding_mode),
      static_cast<int32_t>(rgb_activation),
      static_cast<int32_t>(alpha_activation)};
  return launch_render(
      v,
      vi,
      uv,
      nullptr,
      opacity,
      background,
      static_cast<int32_t>(height),
      static_cast<int32_t>(width),
      static_cast<float>(transmittance_epsilon),
      fragment_evaluator,
      depth_options,
      save_transmittance);
}

std::vector<torch::Tensor> transparent_render_with_mlp_cuda_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& reference_uv,
    const torch::Tensor& texture,
    const torch::Tensor& reference_texture,
    const torch::TensorList& weights,
    const torch::TensorList& biases,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    double negative_slope,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    const TransparentDepthOptions& depth_options,
    bool save_transmittance) {
  check_common_inputs(
      v,
      vi,
      uv,
      opacity,
      background,
      height,
      width,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation);
  check_cuda_float_contiguous(reference_uv, "uv1");
  TORCH_CHECK(
      reference_uv.sizes() == uv.sizes(),
      "transparent render MLP: uv1 must have the same shape as uv0");
  check_texture(texture, v.size(0), "texture0");
  check_texture(reference_texture, v.size(0), "texture1", true);
  const int64_t input_channels = texture.size(1) + reference_texture.size(1);
  TORCH_CHECK(
      input_channels > 0 && input_channels <= kMaxFeatureChannels,
      "transparent render MLP: total feature channels must be in [1, ",
      kMaxFeatureChannels,
      "], but got ",
      input_channels);
  TORCH_CHECK(
      weights.size() == biases.size() && (weights.size() == 4 || weights.size() == 5),
      "transparent render MLP: expected exactly four or five weight/bias pairs");

  TinyMlpParams mlp{};
  mlp.input_channels = static_cast<int32_t>(input_channels);
  mlp.negative_slope = static_cast<float>(negative_slope);
  mlp.rgb_activation = static_cast<int32_t>(rgb_activation);
  mlp.alpha_activation = static_cast<int32_t>(alpha_activation);
  int64_t expected_input = input_channels;
  for (int64_t layer = 0; layer < weights.size(); ++layer) {
    check_cuda_float_contiguous(weights[layer], "MLP weight");
    check_cuda_float_contiguous(biases[layer], "MLP bias");
    const int64_t expected_output = layer == weights.size() - 1 ? 4 : kMlpWidth;
    TORCH_CHECK(
        weights[layer].dim() == 2 && weights[layer].size(0) == expected_output &&
            weights[layer].size(1) == expected_input,
        "transparent render MLP: invalid weight shape at layer ",
        layer);
    TORCH_CHECK(
        biases[layer].dim() == 1 && biases[layer].size(0) == expected_output,
        "transparent render MLP: invalid bias shape at layer ",
        layer);
    TORCH_CHECK(
        weights[layer].device() == v.device() && biases[layer].device() == v.device(),
        "transparent render MLP: all parameters must share the input device");
    mlp.weights[layer] = weights[layer].data_ptr<float>();
    mlp.biases[layer] = biases[layer].data_ptr<float>();
    expected_input = expected_output;
  }

  TORCH_CHECK(
      texture.device() == v.device() && reference_texture.device() == v.device(),
      "transparent render MLP: all inputs must share a device");
  if (weights.size() == 4) {
    MlpFragmentEvaluator<4> fragment_evaluator{
        texture.data_ptr<float>(),
        reference_texture.data_ptr<float>(),
        static_cast<int32_t>(texture.size(0)),
        static_cast<int32_t>(reference_texture.size(0)),
        static_cast<int32_t>(texture.size(1)),
        static_cast<int32_t>(reference_texture.size(1)),
        static_cast<int32_t>(texture.size(2)),
        static_cast<int32_t>(texture.size(3)),
        static_cast<int32_t>(reference_texture.size(2)),
        static_cast<int32_t>(reference_texture.size(3)),
        static_cast<int32_t>(interpolation_mode),
        static_cast<int32_t>(padding_mode),
        mlp};
    return launch_render(
        v,
        vi,
        uv,
        &reference_uv,
        opacity,
        background,
        static_cast<int32_t>(height),
        static_cast<int32_t>(width),
        static_cast<float>(transmittance_epsilon),
        fragment_evaluator,
        depth_options,
        save_transmittance);
  }
  MlpFragmentEvaluator<5> fragment_evaluator{
      texture.data_ptr<float>(),
      reference_texture.data_ptr<float>(),
      static_cast<int32_t>(texture.size(0)),
      static_cast<int32_t>(reference_texture.size(0)),
      static_cast<int32_t>(texture.size(1)),
      static_cast<int32_t>(reference_texture.size(1)),
      static_cast<int32_t>(texture.size(2)),
      static_cast<int32_t>(texture.size(3)),
      static_cast<int32_t>(reference_texture.size(2)),
      static_cast<int32_t>(reference_texture.size(3)),
      static_cast<int32_t>(interpolation_mode),
      static_cast<int32_t>(padding_mode),
      mlp};
  return launch_render(
      v,
      vi,
      uv,
      &reference_uv,
      opacity,
      background,
      static_cast<int32_t>(height),
      static_cast<int32_t>(width),
      static_cast<float>(transmittance_epsilon),
      fragment_evaluator,
      depth_options,
      save_transmittance);
}

std::vector<torch::Tensor> transparent_render_with_texture_cuda_backward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& texture,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& sorted_face_ids,
    const torch::Tensor& saved_transmittance,
    const torch::Tensor& saved_transmittance_exponent,
    const torch::Tensor& grad_color,
    const torch::Tensor& grad_alpha,
    int64_t interpolation_mode,
    int64_t padding_mode,
    int64_t rgb_activation,
    int64_t alpha_activation,
    const torch::Tensor& saved_alpha,
    const torch::Tensor& saved_depth,
    const torch::Tensor& saved_median_face,
    const torch::Tensor& grad_depth,
    const TransparentDepthOptions& depth_options,
    bool edge_gradients) {
  check_backward_inputs(
      v,
      vi,
      uv,
      opacity,
      background,
      tile_offsets,
      sorted_face_ids,
      saved_transmittance,
      saved_transmittance_exponent,
      grad_color,
      grad_alpha,
      interpolation_mode,
      padding_mode,
      rgb_activation,
      alpha_activation);
  check_texture(texture, v.size(0), "texture");
  TORCH_CHECK(
      texture.size(1) == 3 || texture.size(1) == 4,
      "transparent render backward: render_with_texture requires RGB or RGBA input");
  auto texture_grad = torch::zeros_like(texture);
  TextureFragmentBackward fragment_evaluator{
      TextureFragmentEvaluator{
          texture.data_ptr<float>(),
          static_cast<int32_t>(texture.size(0)),
          static_cast<int32_t>(texture.size(1)),
          static_cast<int32_t>(texture.size(2)),
          static_cast<int32_t>(texture.size(3)),
          static_cast<int32_t>(interpolation_mode),
          static_cast<int32_t>(padding_mode),
          static_cast<int32_t>(rgb_activation),
          static_cast<int32_t>(alpha_activation)},
      texture_grad.data_ptr<float>()};
  auto common_grads = launch_render_backward(
      v,
      vi,
      uv,
      nullptr,
      opacity,
      background,
      tile_offsets,
      sorted_face_ids,
      saved_transmittance,
      saved_transmittance_exponent,
      grad_color,
      grad_alpha,
      fragment_evaluator,
      saved_alpha,
      saved_depth,
      saved_median_face,
      grad_depth,
      depth_options,
      edge_gradients);
  return {common_grads[0], common_grads[1], texture_grad, common_grads[3], common_grads[4]};
}

std::vector<torch::Tensor> transparent_render_with_mlp_cuda_backward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& reference_uv,
    const torch::Tensor& texture,
    const torch::Tensor& reference_texture,
    const torch::TensorList& weights,
    const torch::TensorList& biases,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& sorted_face_ids,
    const torch::Tensor& saved_transmittance,
    const torch::Tensor& saved_transmittance_exponent,
    const torch::Tensor& grad_color,
    const torch::Tensor& grad_alpha,
    double negative_slope,
    int64_t interpolation_mode,
    int64_t padding_mode,
    int64_t rgb_activation,
    int64_t alpha_activation,
    const torch::Tensor& saved_alpha,
    const torch::Tensor& saved_depth,
    const torch::Tensor& saved_median_face,
    const torch::Tensor& grad_depth,
    const TransparentDepthOptions& depth_options,
    bool edge_gradients) {
  check_backward_inputs(
      v,
      vi,
      uv,
      opacity,
      background,
      tile_offsets,
      sorted_face_ids,
      saved_transmittance,
      saved_transmittance_exponent,
      grad_color,
      grad_alpha,
      interpolation_mode,
      padding_mode,
      rgb_activation,
      alpha_activation);
  check_cuda_float_contiguous(reference_uv, "uv1");
  TORCH_CHECK(
      reference_uv.sizes() == uv.sizes(), "transparent render MLP backward: uv1 must match uv0");
  check_texture(texture, v.size(0), "texture0");
  check_texture(reference_texture, v.size(0), "texture1", true);
  const int64_t input_channels = texture.size(1) + reference_texture.size(1);
  TORCH_CHECK(
      input_channels <= kMaxFeatureChannels,
      "transparent render MLP backward: too many feature channels");
  TORCH_CHECK(
      weights.size() == biases.size() && (weights.size() == 4 || weights.size() == 5),
      "transparent render MLP backward: expected four or five weight/bias pairs");

  TinyMlpParams mlp{};
  TinyMlpGradParams mlp_grad{};
  mlp.input_channels = static_cast<int32_t>(input_channels);
  mlp.negative_slope = static_cast<float>(negative_slope);
  mlp.rgb_activation = static_cast<int32_t>(rgb_activation);
  mlp.alpha_activation = static_cast<int32_t>(alpha_activation);
  std::vector<torch::Tensor> weight_grads;
  std::vector<torch::Tensor> bias_grads;
  int64_t expected_input = input_channels;
  for (int64_t layer = 0; layer < weights.size(); ++layer) {
    check_cuda_float_contiguous(weights[layer], "MLP weight");
    check_cuda_float_contiguous(biases[layer], "MLP bias");
    const int64_t expected_output = layer == weights.size() - 1 ? 4 : kMlpWidth;
    TORCH_CHECK(
        weights[layer].dim() == 2 && weights[layer].size(0) == expected_output &&
            weights[layer].size(1) == expected_input,
        "transparent render MLP backward: invalid weight shape at layer ",
        layer);
    TORCH_CHECK(
        biases[layer].dim() == 1 && biases[layer].size(0) == expected_output,
        "transparent render MLP backward: invalid bias shape at layer ",
        layer);
    weight_grads.push_back(torch::zeros_like(weights[layer]));
    bias_grads.push_back(torch::zeros_like(biases[layer]));
    mlp.weights[layer] = weights[layer].data_ptr<float>();
    mlp.biases[layer] = biases[layer].data_ptr<float>();
    mlp_grad.weights[layer] = weight_grads.back().data_ptr<float>();
    mlp_grad.biases[layer] = bias_grads.back().data_ptr<float>();
    expected_input = expected_output;
  }

  auto texture_grad = torch::zeros_like(texture);
  auto reference_texture_grad = torch::zeros_like(reference_texture);
  std::vector<torch::Tensor> common_grads;
  if (weights.size() == 4) {
    MlpFragmentBackward<4> fragment_evaluator{
        MlpFragmentEvaluator<4>{
            texture.data_ptr<float>(),
            reference_texture.data_ptr<float>(),
            static_cast<int32_t>(texture.size(0)),
            static_cast<int32_t>(reference_texture.size(0)),
            static_cast<int32_t>(texture.size(1)),
            static_cast<int32_t>(reference_texture.size(1)),
            static_cast<int32_t>(texture.size(2)),
            static_cast<int32_t>(texture.size(3)),
            static_cast<int32_t>(reference_texture.size(2)),
            static_cast<int32_t>(reference_texture.size(3)),
            static_cast<int32_t>(interpolation_mode),
            static_cast<int32_t>(padding_mode),
            mlp},
        texture_grad.data_ptr<float>(),
        reference_texture_grad.data_ptr<float>(),
        mlp_grad,
        mlp_grad};
    common_grads = launch_render_backward(
        v,
        vi,
        uv,
        &reference_uv,
        opacity,
        background,
        tile_offsets,
        sorted_face_ids,
        saved_transmittance,
        saved_transmittance_exponent,
        grad_color,
        grad_alpha,
        fragment_evaluator,
        saved_alpha,
        saved_depth,
        saved_median_face,
        grad_depth,
        depth_options,
        edge_gradients);
  } else {
    MlpFragmentBackward<5> fragment_evaluator{
        MlpFragmentEvaluator<5>{
            texture.data_ptr<float>(),
            reference_texture.data_ptr<float>(),
            static_cast<int32_t>(texture.size(0)),
            static_cast<int32_t>(reference_texture.size(0)),
            static_cast<int32_t>(texture.size(1)),
            static_cast<int32_t>(reference_texture.size(1)),
            static_cast<int32_t>(texture.size(2)),
            static_cast<int32_t>(texture.size(3)),
            static_cast<int32_t>(reference_texture.size(2)),
            static_cast<int32_t>(reference_texture.size(3)),
            static_cast<int32_t>(interpolation_mode),
            static_cast<int32_t>(padding_mode),
            mlp},
        texture_grad.data_ptr<float>(),
        reference_texture_grad.data_ptr<float>(),
        mlp_grad,
        mlp_grad};
    common_grads = launch_render_backward(
        v,
        vi,
        uv,
        &reference_uv,
        opacity,
        background,
        tile_offsets,
        sorted_face_ids,
        saved_transmittance,
        saved_transmittance_exponent,
        grad_color,
        grad_alpha,
        fragment_evaluator,
        saved_alpha,
        saved_depth,
        saved_median_face,
        grad_depth,
        depth_options,
        edge_gradients);
  }

  std::vector<torch::Tensor> result = {
      common_grads[0],
      common_grads[1],
      common_grads[2],
      texture_grad,
      reference_texture_grad,
      common_grads[3],
      common_grads[4]};
  result.insert(result.end(), weight_grads.begin(), weight_grads.end());
  result.insert(result.end(), bias_grads.begin(), bias_grads.end());
  return result;
}

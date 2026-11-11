// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>

#include <vector>

enum class TransparentDepthMode : int32_t {
  Expected = 0,
  Accumulated = 1,
  ForegroundMean = 2,
  Median = 3
};

struct TransparentDepthOptions {
  TransparentDepthMode mode;
  float background;
  float alpha_threshold;
};

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
    bool save_transmittance = true);

std::vector<torch::Tensor> transparent_render_with_mlp_cuda_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv0,
    const torch::Tensor& uv1,
    const torch::Tensor& texture0,
    const torch::Tensor& texture1,
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
    bool save_transmittance = true);

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
    const torch::Tensor& saved_opaque_count,
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
    bool edge_gradients);

std::vector<torch::Tensor> transparent_render_with_mlp_cuda_backward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv0,
    const torch::Tensor& uv1,
    const torch::Tensor& texture0,
    const torch::Tensor& texture1,
    const torch::TensorList& weights,
    const torch::TensorList& biases,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& sorted_face_ids,
    const torch::Tensor& saved_transmittance,
    const torch::Tensor& saved_transmittance_exponent,
    const torch::Tensor& saved_opaque_count,
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
    bool edge_gradients);

// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the MIT license in the root LICENSE file.
#pragma once

#include <torch/types.h>
#include <vector>

struct SourceShadingOptions {
  bool mlp;
  float negative_slope;
  int32_t interpolation_mode;
  int32_t padding_mode;
  int32_t alpha_activation;
};

std::vector<torch::Tensor> source_stencil_appearance_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& query_faces,
    const torch::Tensor& query_bary,
    const torch::Tensor& opacity,
    const torch::TensorList& appearance,
    int64_t height,
    int64_t width,
    const SourceShadingOptions& options,
    int64_t tile_capacity);
std::vector<torch::Tensor> source_stencil_appearance_backward(
    const torch::TensorList& saved,
    const torch::TensorList& grads,
    const torch::TensorList& appearance,
    int64_t height,
    int64_t width,
    bool edge_gradients,
    const SourceShadingOptions& options);

std::vector<torch::Tensor> source_stencil_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& query_faces,
    const torch::Tensor& query_bary,
    const torch::Tensor& opacity,
    int64_t height,
    int64_t width,
    int64_t tile_capacity);
std::vector<torch::Tensor> source_stencil_backward(
    const torch::TensorList& saved,
    const torch::TensorList& grads,
    int64_t height,
    int64_t width,
    bool edge_gradients);
std::vector<torch::Tensor> source_sample_forward(
    const torch::Tensor& features,
    const torch::Tensor& indices,
    const torch::Tensor& weights);
std::vector<torch::Tensor> source_sample_backward(
    const torch::Tensor& features,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    const torch::Tensor& grad_samples,
    const torch::Tensor& grad_confidence);

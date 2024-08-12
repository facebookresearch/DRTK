// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/torch.h>

torch::Tensor grid_scatter_2d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& grid,
    int64_t output_height,
    int64_t output_width,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners);

std::tuple<torch::Tensor, torch::Tensor> grid_scatter_2d_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& grid,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool grid_requires_grad,
    bool input_requires_grad);

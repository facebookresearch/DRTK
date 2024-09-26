// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/torch.h>

torch::Tensor msi_forward_cuda(
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& texture,
    int64_t sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh);

torch::Tensor msi_backward_cuda(
    const torch::Tensor& rgba_img,
    const torch::Tensor& rgba_img_grad,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& texture,
    int64_t sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh);

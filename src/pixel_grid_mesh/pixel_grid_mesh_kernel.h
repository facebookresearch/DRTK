// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license in the root directory.
#pragma once

#include <torch/types.h>
#include <vector>

std::vector<torch::Tensor> pixel_grid_mesh_forward(
    const torch::Tensor& depth,
    const torch::Tensor& displacement,
    const torch::Tensor& intrinsics,
    const torch::Tensor& alive,
    const torch::Tensor& right,
    const torch::Tensor& down,
    const torch::Tensor& diagonal,
    const torch::Tensor& width,
    bool metadata,
    int64_t vertex_capacity,
    int64_t face_capacity);

std::vector<torch::Tensor> pixel_grid_mesh_backward(
    const std::vector<torch::Tensor>& saved,
    const torch::Tensor& grad_v,
    const torch::Tensor& grad_uv,
    bool grad_intrinsics,
    bool grad_width);

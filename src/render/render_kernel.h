// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

std::vector<torch::Tensor>
render_cuda(const torch::Tensor& v, const torch::Tensor& vi, const torch::Tensor& index_img);

torch::Tensor render_cuda_backward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& grad_depth_img,
    const torch::Tensor& grad_bary_img);

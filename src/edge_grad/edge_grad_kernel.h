// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

torch::Tensor edge_grad_estimator_cuda_backward(
    const torch::Tensor& v_pix,
    const torch::Tensor& img,
    const torch::Tensor& index_img,
    const torch::Tensor& vi,
    const torch::Tensor& grad_outputs,
    double max_dp_dr);

torch::Tensor edge_grad_estimator_cpu_fwd(
    const torch::Tensor& v_pix,
    const torch::Tensor& v_pix_img,
    const torch::Tensor& vi,
    const torch::Tensor& img,
    const torch::Tensor& index_img,
    double max_dp_dr);

torch::Tensor edge_grad_estimator_cpu_backward(
    const torch::Tensor& v_pix,
    const torch::Tensor& img,
    const torch::Tensor& index_img,
    const torch::Tensor& vi,
    const torch::Tensor& grad_outputs,
    double max_dp_dr);

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

torch::Tensor interpolate_cuda(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

std::tuple<torch::Tensor, torch::Tensor> interpolate_cuda_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> interpolation_matrix_cuda(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

torch::Tensor interpolation_matrix_cuda_backward(
    const torch::Tensor& grad_values,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    const torch::Tensor& row_pixels);

torch::Tensor interpolation_normal_matrix_values_cuda(
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t nnz);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolation_normal_matrix_cuda(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t num_vertices);

torch::Tensor interpolation_normal_matrix_values_cuda_backward(
    const torch::Tensor& grad_values,
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

torch::Tensor interpolate_cpu(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

std::tuple<torch::Tensor, torch::Tensor> interpolate_cpu_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> interpolation_matrix_cpu(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

torch::Tensor interpolation_matrix_cpu_backward(
    const torch::Tensor& grad_values,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    const torch::Tensor& row_pixels);

torch::Tensor interpolation_normal_matrix_values_cpu(
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t nnz);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolation_normal_matrix_cpu(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t num_vertices);

torch::Tensor interpolation_normal_matrix_values_cpu_backward(
    const torch::Tensor& grad_values,
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

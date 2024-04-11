#pragma once

torch::Tensor compute_vert_image_cuda(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

std::tuple<torch::Tensor, torch::Tensor> compute_vert_image_cuda_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img);

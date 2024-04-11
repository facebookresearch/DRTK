#pragma once
#include <torch/torch.h>

torch::Tensor msi_bkg_forward_cuda(
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& texture,
    int64_t sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh);

torch::Tensor msi_bkg_backward_cuda(
    const torch::Tensor& rgba_img,
    const torch::Tensor& rgba_img_grad,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& texture,
    int64_t sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh);

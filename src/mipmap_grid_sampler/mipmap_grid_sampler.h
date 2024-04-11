#pragma once
#include <torch/torch.h>

torch::Tensor mipmap_aniso_grid_sampler_2d_cuda(
    const torch::TensorList& input,
    const torch::Tensor& grid,
    const torch::Tensor& vt_dxdy_img,
    int64_t max_aniso,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool force_max_ansio,
    bool clip_grad);

std::tuple<std::vector<torch::Tensor>, torch::Tensor> mipmap_aniso_grid_sampler_2d_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::TensorList& input,
    const torch::Tensor& grid,
    const torch::Tensor& vt_dxdy_img,
    int64_t max_aniso,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool force_max_ansio,
    bool clip_grad);

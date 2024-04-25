#pragma once

torch::Tensor edge_grad_estimator_cuda_backward(
    const torch::Tensor& v_pix,
    const torch::Tensor& img,
    const torch::Tensor& index_img,
    const torch::Tensor& vi,
    const torch::Tensor& grad_outputs);

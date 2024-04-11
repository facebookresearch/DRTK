#pragma once

#include <torch/torch.h>

torch::Tensor edge_grad_estimator_autograd(
    const torch::Tensor v_pix,
    const torch::Tensor v_pix_img,
    const torch::Tensor vi,
    const torch::Tensor img,
    const torch::Tensor index_img);

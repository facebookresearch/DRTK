#pragma once

#include <torch/torch.h>
#include <vector>
#include "../include/common.h"

std::vector<torch::Tensor> render_forward(
    torch::Tensor v2d,
    torch::Tensor vt,
    at::optional<torch::Tensor> vn,
    torch::Tensor vi,
    torch::Tensor vti,
    torch::Tensor indeximg,
    torch::Tensor depthimg,
    torch::Tensor baryimg,
    torch::Tensor uvimg,
    at::optional<torch::Tensor> vnimg);

std::vector<torch::Tensor> render_backward(
    torch::Tensor v2d,
    torch::Tensor vt,
    at::optional<torch::Tensor> vn,
    torch::Tensor vi,
    torch::Tensor vti,
    torch::Tensor indeximg,
    torch::Tensor grad_depthimg,
    torch::Tensor grad_baryimg,
    torch::Tensor grad_uvimg,
    at::optional<torch::Tensor> grad_vnimg,
    torch::Tensor grad_v2d,
    at::optional<torch::Tensor> grad_vn);

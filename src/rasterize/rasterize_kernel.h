#pragma once

std::vector<torch::Tensor> rasterize_cuda(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    int64_t height,
    int64_t width,
    bool wireframe);

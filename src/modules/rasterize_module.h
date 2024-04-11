#pragma once

#include <torch/torch.h>

int64_t rasterize_packed(
    torch::Tensor verts_t,
    torch::Tensor vind_t,
    torch::Tensor depth_img_t,
    torch::Tensor index_img_t,
    torch::Tensor packedindex_img_t);

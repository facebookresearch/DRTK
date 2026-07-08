#pragma once
#include <torch/torch.h>

torch::Tensor
filter2d_fused(torch::Tensor x, torch::Tensor f, int up, int down, bool backward, bool reflect);

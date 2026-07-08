#pragma once
#include <torch/torch.h>

enum class FilterType : int64_t { Kaiser = 0, Lanczos = 1 };

torch::Tensor make_resampling_kernel(
    int64_t n,
    int64_t m,
    double freq_div,
    double gain,
    double strength,
    int64_t filter_type,
    torch::Device device);

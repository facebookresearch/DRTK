#include <torch/torch.h>

#include "filter_weights.h"
#include "impls.h"

class ResampleFilterFunction : public torch::autograd::Function<ResampleFilterFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor x,
      const torch::Tensor f,
      int64_t up,
      int64_t down,
      bool backward_flag,
      bool reflect) {
    ctx->set_materialize_grads(false);
    ctx->save_for_backward({f});
    ctx->saved_data["data"] = std::make_tuple(x.requires_grad(), up, down, backward_flag, reflect);

    return {filter2d_fused(x, f, up, down, backward_flag, reflect)};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    bool x_requires_grad;
    int64_t up;
    int64_t down;
    bool backward_flag;
    bool reflect;
    std::tie(x_requires_grad, up, down, backward_flag, reflect) =
        ctx->saved_data["data"].to<std::tuple<bool, int64_t, int64_t, bool, bool>>();
    if (!x_requires_grad) {
      return {
          torch::Tensor(),
          torch::Tensor(),
          torch::Tensor(),
          torch::Tensor(),
          torch::Tensor(),
          torch::Tensor()};
    }
    const auto saved = ctx->get_saved_variables();
    const auto f = saved[0];
    const auto grad_output = grad_outputs[0].contiguous();
    const auto x_grad =
        ResampleFilterFunction::apply(grad_output, f, down, up, !backward_flag, reflect)[0];

    return {
        x_grad,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()};
  }
};

torch::Tensor resample_filter_autograd(
    const torch::Tensor x,
    const torch::Tensor f,
    int64_t up,
    int64_t down,
    bool reflect) {
  return ResampleFilterFunction::apply(x, f, up, down, false, reflect)[0];
}

torch::Tensor
resample_filter_fwd(torch::Tensor x, torch::Tensor f, int64_t up, int64_t down, bool reflect) {
  return filter2d_fused(x, f, up, down, false, reflect);
}

torch::Tensor low_pass_filter_fwd(
    torch::Tensor tensor,
    int64_t n,
    double freq_div,
    double alias_guard_band,
    int64_t filter_type,
    bool reflect) {
  torch::Tensor filter =
      make_resampling_kernel(n, 1, freq_div, 1.0, alias_guard_band, filter_type, tensor.device());
  return resample_filter_fwd(tensor, filter, 1, 1, reflect);
}

torch::Tensor low_pass_filter_autograd(
    torch::Tensor tensor,
    int64_t n,
    double freq_div,
    double alias_guard_band,
    int64_t filter_type,
    bool reflect) {
  torch::Tensor filter =
      make_resampling_kernel(n, 1, freq_div, 1.0, alias_guard_band, filter_type, tensor.device());
  return resample_filter_autograd(tensor, filter, 1, 1, reflect);
}

torch::Tensor downsample_fwd(
    torch::Tensor tensor,
    int64_t n,
    int64_t m,
    double alias_guard_band,
    int64_t filter_type,
    bool reflect) {
  torch::Tensor filter =
      make_resampling_kernel(n, m, 1.0, 1.0, alias_guard_band, filter_type, tensor.device());
  return resample_filter_fwd(tensor, filter, 1, m, reflect);
}

torch::Tensor downsample_autograd(
    torch::Tensor tensor,
    int64_t n,
    int64_t m,
    double alias_guard_band,
    int64_t filter_type,
    bool reflect) {
  torch::Tensor filter =
      make_resampling_kernel(n, m, 1.0, 1.0, alias_guard_band, filter_type, tensor.device());
  return resample_filter_autograd(tensor, filter, 1, m, reflect);
}

torch::Tensor upsample_fwd(
    torch::Tensor tensor,
    int64_t n,
    int64_t m,
    double alias_guard_band,
    int64_t filter_type,
    bool reflect) {
  torch::Tensor filter =
      make_resampling_kernel(n, m, 1.0, m, alias_guard_band, filter_type, tensor.device());
  return resample_filter_fwd(tensor, filter, m, 1, reflect);
}

torch::Tensor upsample_autograd(
    torch::Tensor tensor,
    int64_t n,
    int64_t m,
    double alias_guard_band,
    int64_t filter_type,
    bool reflect) {
  torch::Tensor filter =
      make_resampling_kernel(n, m, 1.0, m, alias_guard_band, filter_type, tensor.device());
  return resample_filter_autograd(tensor, filter, m, 1, reflect);
}

// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
#ifndef NO_PYBIND
PYBIND11_MODULE(filter2d_ext, m) {
  m.def("resample_filter", &resample_filter_autograd);
  m.def("low_pass_filter", &low_pass_filter_autograd);
  m.def("downsample", &downsample_autograd);
  m.def("upsample", &upsample_autograd);
  m.def("make_resampling_kernel", &make_resampling_kernel);
}
#endif

TORCH_LIBRARY(filter2d_ext, m) {
  m.def("resample_filter(Tensor x, Tensor f, int up, int down, bool reflect) -> Tensor");
  m.def(
      "low_pass_filter(Tensor x, int n, float freq_div, float alias_guard_band, int filter_type, bool reflect) -> Tensor");
  m.def(
      "downsample(Tensor x, int n, int m, float alias_guard_band, int filter_type, bool reflect) -> Tensor");
  m.def(
      "upsample(Tensor x, int n, int m, float alias_guard_band, int filter_type, bool reflect) -> Tensor");
  m.def(
      "make_resampling_kernel(int n, int m, float freq_div, float gain, float alias_guard_band, int filter_type, Device d) -> Tensor",
      &make_resampling_kernel);
}

TORCH_LIBRARY_IMPL(filter2d_ext, Autograd, m) {
  m.impl("resample_filter", &resample_filter_autograd);
  m.impl("low_pass_filter", &low_pass_filter_autograd);
  m.impl("downsample", &downsample_autograd);
  m.impl("upsample", &upsample_autograd);
}

TORCH_LIBRARY_IMPL(filter2d_ext, CUDA, m) {
  m.impl("resample_filter", &resample_filter_fwd);
  m.impl("low_pass_filter", &low_pass_filter_fwd);
  m.impl("downsample", &downsample_fwd);
  m.impl("upsample", &upsample_fwd);
}

TORCH_LIBRARY_IMPL(filter2d_ext, CPU, m) {
  m.impl("resample_filter", &resample_filter_fwd);
  m.impl("low_pass_filter", &low_pass_filter_fwd);
  m.impl("downsample", &downsample_fwd);
  m.impl("upsample", &upsample_fwd);
}

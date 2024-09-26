// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/autocast_mode.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "rasterize_kernel.h"

// Dispatch function
torch::autograd::tensor_list rasterize(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    int64_t height,
    int64_t width,
    bool wireframe) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("rasterize_ext::rasterize", "")
                       .typed<decltype(rasterize)>();
  return op.call(v, vi, height, width, wireframe);
}

// Ideally we would need to turn off autograd handling and re-dispatch, but we just call
// cuda kernels directly
class RasterizeFunction : public torch::autograd::Function<RasterizeFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& v,
      const torch::Tensor& vi,
      int64_t height,
      int64_t width,
      bool wireframe) {
    ctx->set_materialize_grads(false);
    auto outputs = rasterize_cuda(v, vi, height, width, wireframe);
    ctx->mark_non_differentiable(outputs);
    return outputs;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::tensor_list& grad_outputs) {
    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::autograd::tensor_list rasterize_autograd(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    int64_t height,
    int64_t width,
    bool wireframe) {
  return RasterizeFunction::apply(v, vi, height, width, wireframe);
}

torch::autograd::tensor_list rasterize_autocast(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    int64_t height,
    int64_t width,
    bool wireframe) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return rasterize(at::autocast::cached_cast(torch::kFloat32, v), vi, height, width, wireframe);
}

#ifndef NO_PYBIND
PYBIND11_MODULE(rasterize_ext, m) {}
#endif

TORCH_LIBRARY(rasterize_ext, m) {
  m.def("rasterize(Tensor v, Tensor vi, int height, int width, bool wireframe) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rasterize_ext, Autograd, m) {
  m.impl("rasterize", &rasterize_autograd);
}

TORCH_LIBRARY_IMPL(rasterize_ext, Autocast, m) {
  m.impl("rasterize", rasterize_autocast);
}

TORCH_LIBRARY_IMPL(rasterize_ext, CUDA, m) {
  m.impl("rasterize", &rasterize_cuda);
}

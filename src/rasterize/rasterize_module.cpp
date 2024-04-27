#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "rasterize_kernel.h"

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
      torch::autograd::tensor_list grad_outputs) {
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

#ifndef NO_PYBIND
PYBIND11_MODULE(rasterize_ext, m) {}
#endif

TORCH_LIBRARY(rasterize_ext, m) {
  m.def("rasterize(Tensor v, Tensor vi, int height, int width, bool wireframe) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rasterize_ext, Autograd, m) {
  m.impl("rasterize", &rasterize_autograd);
}

TORCH_LIBRARY_IMPL(rasterize_ext, CUDA, m) {
  m.impl("rasterize", &rasterize_cuda);
}

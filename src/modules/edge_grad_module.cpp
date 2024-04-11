#include <torch/script.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "../edge_grad/edge_grad_kernel.h"

torch::Tensor edge_grad_estimator_fwd(
    const torch::Tensor v_pix,
    const torch::Tensor v_pix_img,
    const torch::Tensor vi,
    const torch::Tensor img,
    const torch::Tensor index_img) {
  return img;
}

#ifndef NO_PYBIND
// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
PYBIND11_MODULE(edge_grad_ext, m) {
  m.def("edge_grad_estimator", &edge_grad_estimator_fwd);
}
#endif

TORCH_LIBRARY(edge_grad_ext, m) {
  m.def(
      "edge_grad_estimator(Tensor v_pix, Tensor v_pix_img, Tensor vi, Tensor img, Tensor index_img) -> Tensor");
}

TORCH_LIBRARY_IMPL(edge_grad_ext, Autograd, m) {
  m.impl("edge_grad_estimator", &edge_grad_estimator_autograd);
}

TORCH_LIBRARY_IMPL(edge_grad_ext, CUDA, m) {
  m.impl("edge_grad_estimator", &edge_grad_estimator_fwd);
}

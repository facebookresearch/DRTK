// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/autocast_mode.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "edge_grad_kernel.h"

// Dispatch function
torch::Tensor edge_grad_estimator(
    const torch::Tensor& v_pix,
    const torch::Tensor& v_pix_img,
    const torch::Tensor& vi,
    const torch::Tensor& img,
    const torch::Tensor& index_img) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("edge_grad_ext::edge_grad_estimator", "")
                       .typed<decltype(edge_grad_estimator)>();
  return op.call(v_pix, v_pix_img, vi, img, index_img);
}

torch::Tensor edge_grad_estimator_fwd(
    const torch::Tensor& v_pix,
    const torch::Tensor& v_pix_img,
    const torch::Tensor& vi,
    const torch::Tensor& img,
    const torch::Tensor& index_img) {
  TORCH_CHECK(
      v_pix.defined() && v_pix_img.defined() && vi.defined() && img.defined() &&
          index_img.defined(),
      "edge_grad_estimator(): expected all inputs to be defined");
  TORCH_CHECK(
      (v_pix.device() == v_pix_img.device()) && (v_pix.device() == vi.device()) &&
          (v_pix.device() == img.device()) && (v_pix.device() == index_img.device()) &&
          (v_pix.is_cuda()),
      "edge_grad_estimator(): expected all inputs to be on same cuda device");
  TORCH_CHECK(
      v_pix.is_floating_point() && v_pix_img.is_floating_point() && img.is_floating_point(),
      "edge_grad_estimator(): expected v_pix, v_pix_img, and img to have floating point type, but v_pix has ",
      v_pix.dtype(),
      " v_pix has ",
      v_pix_img.dtype(),
      " img has ",
      img.dtype());
  TORCH_CHECK(
      vi.dtype() == torch::kInt32,
      "edge_grad_estimator(): expected vi to have int32 type, but vi has ",
      vi.dtype());
  TORCH_CHECK(
      index_img.dtype() == torch::kInt32,
      "edge_grad_estimator(): expected index_img to have int32 type, but index_img has ",
      index_img.dtype());
  TORCH_CHECK(
      v_pix.layout() == torch::kStrided && v_pix_img.layout() == torch::kStrided &&
          vi.layout() == torch::kStrided && img.layout() == torch::kStrided &&
          index_img.layout() == torch::kStrided,
      "edge_grad_estimator(): expected all inputs to have torch.strided layout");
  TORCH_CHECK(
      (v_pix.dim() == 3) && (v_pix_img.dim() == 4) && (vi.dim() == 2) && (img.dim() == 4) &&
          (index_img.dim() == 3),
      "edge_grad_estimator(): expected v_pix.ndim == 3, v_pix_img.ndim == 4, vi.ndim == 2, img.ndim == 4, index_img.ndim == 3, "
      "but got v_pix with sizes ",
      v_pix.sizes(),
      " and v_pix_img with sizes ",
      v_pix_img.sizes(),
      " and vi with sizes ",
      vi.sizes(),
      " and img with sizes ",
      img.sizes(),
      " and index_img with sizes ",
      index_img.sizes());
  TORCH_CHECK(
      v_pix.size(0) == v_pix_img.size(0) && v_pix.size(0) == img.size(0) &&
          v_pix.size(0) == index_img.size(0),
      "edge_grad_estimator(): expected v and index_img to have same batch size, "
      "but got v_pix with sizes ",
      v_pix.sizes(),
      ", v_pix_img with sizes ",
      v_pix_img.sizes(),
      ", img with sizes ",
      img.sizes(),
      " and index_img with sizes ",
      index_img.sizes());
  TORCH_CHECK(
      v_pix.size(2) == 3 && v_pix_img.size(1) == 3 && vi.size(1) == 3,
      "edge_grad_estimator(): expected third dim of v_pix to be of size 3, and second dim of vi to be of size 3, but got ",
      v_pix.size(2),
      " in the third dim of v_pix, and ",
      v_pix_img.size(1),
      " in the second dim of v_pix_img, and ",
      vi.size(1),
      " in the second dim of vi");
  TORCH_CHECK(
      v_pix_img.size(3) == img.size(3) && v_pix_img.size(3) == index_img.size(2) &&
          v_pix_img.size(2) == img.size(2) && v_pix_img.size(2) == index_img.size(1),
      "edge_grad_estimator(): expected width and height of v_pix_img, img, and index_img to match, but got size of v_pix_img: ",
      v_pix_img.sizes(),
      ", size of img: ",
      img.sizes(),
      ", size of index_img: ",
      index_img.sizes());
  return img;
}

// Ideally we would need to turn off autograd handling and re-dispatch, but we just call
// cuda kernels directly
class EdgeGradEstimatorFunction : public torch::autograd::Function<EdgeGradEstimatorFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& v_pix,
      const torch::Tensor& v_pix_img,
      const torch::Tensor& vi,
      const torch::Tensor& img,
      const torch::Tensor& index_img) {
    ctx->set_materialize_grads(false);
    ctx->save_for_backward({v_pix, img, index_img, vi});
    ctx->saved_data["v_pix_img_requires_grad"] = v_pix_img.requires_grad();
    return {img};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    // If v_pix_img doesn't require grad, we don't need to do anything.
    if (!ctx->saved_data["v_pix_img_requires_grad"].toBool()) {
      return {torch::Tensor(), torch::Tensor(), torch::Tensor(), grad_outputs[0], torch::Tensor()};
    }
    const auto saved = ctx->get_saved_variables();
    const auto& v_pix = saved[0];
    const auto& img = saved[1];
    const auto& index_img = saved[2];
    const auto& vi = saved[3];

    auto grad_v_pix_img =
        edge_grad_estimator_cuda_backward(v_pix, img, index_img, vi, grad_outputs[0]);
    return {torch::Tensor(), grad_v_pix_img, torch::Tensor(), grad_outputs[0], torch::Tensor()};
  }
};

torch::Tensor edge_grad_estimator_autograd(
    const torch::Tensor& v_pix,
    const torch::Tensor& v_pix_img,
    const torch::Tensor& vi,
    const torch::Tensor& img,
    const torch::Tensor& index_img) {
  return EdgeGradEstimatorFunction::apply(v_pix, v_pix_img, vi, img, index_img)[0];
}

torch::Tensor edge_grad_estimator_autocast(
    const torch::Tensor& v_pix,
    const torch::Tensor& v_pix_img,
    const torch::Tensor& vi,
    const torch::Tensor& img,
    const torch::Tensor& index_img) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return edge_grad_estimator(
      at::autocast::cached_cast(torch::kFloat32, v_pix),
      at::autocast::cached_cast(torch::kFloat32, v_pix_img),
      vi,
      at::autocast::cached_cast(torch::kFloat32, img),
      index_img)[0];
}

#ifndef NO_PYBIND
// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
PYBIND11_MODULE(edge_grad_ext, m) {}
#endif

TORCH_LIBRARY(edge_grad_ext, m) {
  m.def(
      "edge_grad_estimator(Tensor v_pix, Tensor v_pix_img, Tensor vi, Tensor img, Tensor index_img) -> Tensor");
}

TORCH_LIBRARY_IMPL(edge_grad_ext, Autograd, m) {
  m.impl("edge_grad_estimator", &edge_grad_estimator_autograd);
}

TORCH_LIBRARY_IMPL(edge_grad_ext, Autocast, m) {
  m.impl("edge_grad_estimator", edge_grad_estimator_autocast);
}

TORCH_LIBRARY_IMPL(edge_grad_ext, CUDA, m) {
  m.impl("edge_grad_estimator", &edge_grad_estimator_fwd);
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/autocast_mode.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "interpolate_kernel.h"

// Dispatch function
torch::Tensor interpolate(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("interpolate_ext::interpolate", "")
                       .typed<decltype(interpolate)>();
  return op.call(vert_attributes, vi, index_img, bary_img);
}

// Ideally we would need to turn off autograd handling and re-dispatch, but we just call
// cuda kernels directly
class InterpolateFunction : public torch::autograd::Function<InterpolateFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& vert_attributes,
      const torch::Tensor& vi,
      const torch::Tensor& index_img,
      const torch::Tensor& bary_img) {
    ctx->set_materialize_grads(false);
    std::vector<torch::Tensor> save_list;
    save_list.push_back(vert_attributes);
    save_list.push_back(vi);
    save_list.push_back(index_img);
    save_list.push_back(bary_img);
    ctx->save_for_backward(save_list);
    return {interpolate_cuda(vert_attributes, vi, index_img, bary_img)};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& vert_attributes = saved[0];
    const torch::Tensor& vi = saved[1];
    const torch::Tensor& index_img = saved[2];
    const torch::Tensor& bary_img = saved[3];
    bool bary_img_requires_grad = bary_img.requires_grad();
    bool vert_requires_grad = vert_attributes.requires_grad();

    torch::autograd::tensor_list out;
    if ((!bary_img_requires_grad && !vert_requires_grad) || !grad_outputs[0].defined()) {
      out.resize(4);
      return out;
    }
    auto grad_out =
        interpolate_cuda_backward(grad_outputs[0], vert_attributes, vi, index_img, bary_img);

    out.push_back(std::get<0>(grad_out));
    out.emplace_back();
    out.emplace_back();
    out.push_back(std::get<1>(grad_out));
    return out;
  }
};

torch::Tensor interpolate_autograd(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  return InterpolateFunction::apply(vert_attributes, vi, index_img, bary_img)[0];
}

torch::Tensor interpolate_autocast(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return interpolate(
      at::autocast::cached_cast(torch::kFloat32, vert_attributes),
      vi,
      index_img,
      at::autocast::cached_cast(torch::kFloat32, bary_img));
}

#ifndef NO_PYBIND
// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
PYBIND11_MODULE(interpolate_ext, m) {}
#endif

TORCH_LIBRARY(interpolate_ext, m) {
  m.def(
      "interpolate(Tensor vert_attributes, Tensor vi, Tensor index_img, Tensor bary_img) -> Tensor");
}

TORCH_LIBRARY_IMPL(interpolate_ext, Autograd, m) {
  m.impl("interpolate", &interpolate_autograd);
}

TORCH_LIBRARY_IMPL(interpolate_ext, Autocast, m) {
  m.impl("interpolate", interpolate_autocast);
}

TORCH_LIBRARY_IMPL(interpolate_ext, CUDA, m) {
  m.impl("interpolate", &interpolate_cuda);
}

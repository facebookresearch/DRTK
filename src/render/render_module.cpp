// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/autocast_mode.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "render_kernel.h"

// Dispatch function
torch::autograd::tensor_list
render(const torch::Tensor& v, const torch::Tensor& vi, const torch::Tensor& index_img) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("render_ext::render", "")
                       .typed<decltype(render)>();
  return op.call(v, vi, index_img);
}

// Ideally we would need to turn off autograd handling and re-dispatch, but we just call
// cuda kernels directly
class RenderFunction : public torch::autograd::Function<RenderFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& v,
      const torch::Tensor& vi,
      const torch::Tensor& index_img) {
    // ctx->set_materialize_grads(false);
    std::vector<torch::Tensor> save_list;
    save_list.push_back(v);
    save_list.push_back(vi);
    save_list.push_back(index_img);
    ctx->save_for_backward(save_list);

    ctx->saved_data["data"] = std::make_tuple((bool)v.requires_grad());

    auto outputs = render_cuda(v, vi, index_img);
    return outputs;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& v = saved[0];
    const torch::Tensor& vi = saved[1];
    const torch::Tensor& index_img = saved[2];

    bool requires_grad;
    std::tie(requires_grad) = ctx->saved_data["data"].to<std::tuple<bool>>();

    torch::autograd::tensor_list out;
    if (!requires_grad) {
      out.resize(3);
      return out;
    }
    auto grad_v = render_cuda_backward(v, vi, index_img, grad_outputs[0], grad_outputs[1]);

    out.push_back(grad_v);
    out.emplace_back();
    out.emplace_back();
    return out;
  }
};

torch::autograd::tensor_list
render_autograd(const torch::Tensor& v, const torch::Tensor& vi, const torch::Tensor& index_img) {
  return RenderFunction::apply(v, vi, index_img);
}

torch::autograd::tensor_list
render_autocast(const torch::Tensor& v, const torch::Tensor& vi, const torch::Tensor& index_img) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return render(at::autocast::cached_cast(torch::kFloat32, v), vi, index_img);
}

#ifndef NO_PYBIND
PYBIND11_MODULE(render_ext, m) {}
#endif

TORCH_LIBRARY(render_ext, m) {
  m.def("render(Tensor v, Tensor vi, Tensor index_img) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(render_ext, Autograd, m) {
  m.impl("render", &render_autograd);
}

TORCH_LIBRARY_IMPL(render_ext, Autocast, m) {
  m.impl("render", render_autocast);
}

TORCH_LIBRARY_IMPL(render_ext, CUDA, m) {
  m.impl("render", &render_cuda);
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/autocast_mode.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "grid_scatter_kernel.h"

// Dispatch function
torch::Tensor grid_scatter_2d(
    const torch::Tensor& input,
    const torch::Tensor& grid,
    int64_t output_height,
    int64_t output_width,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("grid_scatter_ext::grid_scatter_2d", "")
                       .typed<decltype(grid_scatter_2d)>();
  return op.call(
      input, grid, output_height, output_width, padding_mode, interpolation_mode, align_corners);
}

// Ideally we would need to turn off autograd handling and re-dispatch, but we just call
// cuda kernels directly
class GridScatter2DFunction : public torch::autograd::Function<GridScatter2DFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& input,
      const torch::Tensor& grid,
      int64_t output_height,
      int64_t output_width,
      int64_t padding_mode,
      int64_t interpolation_mode,
      bool align_corners) {
    ctx->set_materialize_grads(false);
    std::vector<torch::Tensor> save_list;
    save_list.push_back(input);
    save_list.push_back(grid);
    ctx->save_for_backward(save_list);
    bool grid_requires_grad = grid.requires_grad();
    bool input_requires_grad = input.requires_grad();

    ctx->saved_data["data"] = std::make_tuple(
        grid_requires_grad, input_requires_grad, padding_mode, interpolation_mode, align_corners);
    auto out = grid_scatter_2d_cuda(
        input, grid, output_height, output_width, padding_mode, interpolation_mode, align_corners);
    return {out};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    bool grid_requires_grad;
    bool input_requires_grad;
    int64_t padding_mode;
    int64_t interpolation_mode;
    bool align_corners;
    std::tie(
        grid_requires_grad, input_requires_grad, padding_mode, interpolation_mode, align_corners) =
        ctx->saved_data["data"].to<std::tuple<bool, bool, int64_t, int64_t, bool>>();
    torch::autograd::tensor_list out;
    if (!grid_requires_grad && !input_requires_grad) {
      out.resize(7);
      return out;
    }
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& input = saved[0];
    const torch::Tensor& grid = saved[1];
    auto grad_out = grid_scatter_2d_cuda_backward(
        grad_outputs[0],
        input,
        grid,
        padding_mode,
        interpolation_mode,
        align_corners,
        grid_requires_grad,
        input_requires_grad);

    out.push_back(std::get<0>(grad_out));
    out.push_back(std::get<1>(grad_out));
    out.emplace_back();
    out.emplace_back();
    out.emplace_back();
    out.emplace_back();
    out.emplace_back();
    return out;
  }
};

torch::Tensor grid_scatter_2d_autograd(
    const torch::Tensor& input,
    const torch::Tensor& grid,
    int64_t output_height,
    int64_t output_width,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners) {
  return GridScatter2DFunction::apply(
      input, grid, output_height, output_width, padding_mode, interpolation_mode, align_corners)[0];
}

torch::Tensor grid_scatter_2d_autocast(
    const torch::Tensor& input,
    const torch::Tensor& grid,
    int64_t output_height,
    int64_t output_width,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return grid_scatter_2d(
      at::autocast::cached_cast(torch::kFloat32, input),
      at::autocast::cached_cast(torch::kFloat32, grid),
      output_height,
      output_width,
      padding_mode,
      interpolation_mode,
      align_corners);
}

#ifndef NO_PYBIND
// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
PYBIND11_MODULE(grid_scatter_ext, m) {}
#endif

TORCH_LIBRARY(grid_scatter_ext, m) {
  m.def(
      "grid_scatter_2d(Tensor input, Tensor grid, int output_height, int output_width, int padding_mode, int interpolation_mode, bool align_corners) -> Tensor");
}

TORCH_LIBRARY_IMPL(grid_scatter_ext, Autograd, m) {
  m.impl("grid_scatter_2d", &grid_scatter_2d_autograd);
}

TORCH_LIBRARY_IMPL(grid_scatter_ext, Autocast, m) {
  m.impl("grid_scatter_2d", grid_scatter_2d_autocast);
}

TORCH_LIBRARY_IMPL(grid_scatter_ext, CUDA, m) {
  m.impl("grid_scatter_2d", &grid_scatter_2d_cuda);
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license in the root directory.

#include <torch/script.h>
#ifndef NO_PYBIND
#include <torch/extension.h>
#endif
#include "pixel_grid_mesh_kernel.h"

namespace {
using torch::autograd::tensor_list;

class PixelGridMeshFunction : public torch::autograd::Function<PixelGridMeshFunction> {
 public:
  static tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& depth,
      const torch::Tensor& displacement,
      const torch::Tensor& intrinsics,
      const torch::Tensor& alive,
      const torch::Tensor& right,
      const torch::Tensor& down,
      const torch::Tensor& diagonal,
      const torch::Tensor& width,
      bool metadata) {
    auto out = pixel_grid_mesh_forward(
        depth, displacement, intrinsics, alive, right, down, diagonal, width, metadata);
    // Floating inputs, compact vertices, then local topology/geometry state.
    tensor_list saved{depth, displacement, intrinsics, width, out[0]};
    saved.insert(saved.end(), out.begin() + 7, out.begin() + 15);
    ctx->save_for_backward(saved);
    ctx->mark_non_differentiable({out[1], out[3], out[4], out[5], out[6]});
    ctx->set_materialize_grads(false);
    return tensor_list(out.begin(), out.begin() + 7);
  }

  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list outputs) {
    auto gradients = pixel_grid_mesh_backward(
        ctx->get_saved_variables(),
        outputs[0],
        outputs[2],
        ctx->needs_input_grad(2),
        ctx->needs_input_grad(7));
    return {
        gradients[0],
        gradients[1],
        gradients[2],
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        gradients[3],
        torch::Tensor()};
  }
};

tensor_list build(
    const torch::Tensor& depth,
    const torch::Tensor& displacement,
    const torch::Tensor& intrinsics,
    const torch::Tensor& alive,
    const torch::Tensor& right,
    const torch::Tensor& down,
    const torch::Tensor& diagonal,
    const torch::Tensor& width,
    bool metadata) {
  return PixelGridMeshFunction::apply(
      depth, displacement, intrinsics, alive, right, down, diagonal, width, metadata);
}
} // namespace

#ifndef NO_PYBIND
PYBIND11_MODULE(pixel_grid_mesh_ext, m) {}
#endif

TORCH_LIBRARY(pixel_grid_mesh_ext, m) {
  m.def(
      "build(Tensor depth, Tensor displacement, Tensor intrinsics, Tensor alive, "
      "Tensor right, Tensor down, Tensor diagonal, Tensor width, bool metadata) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(pixel_grid_mesh_ext, CompositeImplicitAutograd, m) {
  m.impl("build", &build);
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/autocast_mode.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "mipmap_grid_sampler_kernel.h"

// Dispatch function
torch::Tensor mipmap_grid_sampler_2d(
    const torch::TensorList& input,
    const torch::Tensor& grid,
    const torch::Tensor& vt_dxdy_img,
    int64_t max_aniso,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool force_max_ansio,
    bool clip_grad) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("mipmap_grid_sampler_ext::mipmap_grid_sampler_2d", "")
                       .typed<decltype(mipmap_grid_sampler_2d)>();
  return op.call(
      input,
      grid,
      vt_dxdy_img,
      max_aniso,
      padding_mode,
      interpolation_mode,
      align_corners,
      force_max_ansio,
      clip_grad);
}

// Ideally we would need to turn off autograd handling and re-dispatch, but we just call
// cuda kernels directly
class MipmapGridSample2DFunction : public torch::autograd::Function<MipmapGridSample2DFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& grid,
      const torch::Tensor& vt_dxdy_img,
      int64_t max_aniso,
      int64_t padding_mode,
      int64_t interpolation_mode,
      bool align_corners,
      bool force_max_ansio,
      bool clip_grad,
      const torch::Tensor& input0,
      const c10::optional<torch::Tensor>& input1,
      const c10::optional<torch::Tensor>& input2,
      const c10::optional<torch::Tensor>& input3,
      const c10::optional<torch::Tensor>& input4,
      const c10::optional<torch::Tensor>& input5,
      const c10::optional<torch::Tensor>& input6,
      const c10::optional<torch::Tensor>& input7,
      const c10::optional<torch::Tensor>& input8,
      const c10::optional<torch::Tensor>& input9,
      const c10::optional<torch::Tensor>& input10) {
    std::vector<torch::Tensor> input = {input0};
    if (input1.has_value())
      input.push_back(input1.value());
    if (input2.has_value())
      input.push_back(input2.value());
    if (input3.has_value())
      input.push_back(input3.value());
    if (input4.has_value())
      input.push_back(input4.value());
    if (input5.has_value())
      input.push_back(input5.value());
    if (input6.has_value())
      input.push_back(input6.value());
    if (input7.has_value())
      input.push_back(input7.value());
    if (input8.has_value())
      input.push_back(input8.value());
    if (input9.has_value())
      input.push_back(input9.value());
    if (input10.has_value())
      input.push_back(input10.value());

    ctx->set_materialize_grads(false);
    std::vector<torch::Tensor> save_list;
    for (auto& inp : input) {
      save_list.push_back(inp);
    }
    save_list.push_back(grid);
    save_list.push_back(vt_dxdy_img);
    ctx->save_for_backward(save_list);
    bool requires_grad = false;
    for (const auto& inp : input) {
      requires_grad = requires_grad || inp.requires_grad();
    }
    requires_grad = requires_grad || grid.requires_grad();

    ctx->saved_data["data"] = std::make_tuple(
        (int64_t)input.size(),
        requires_grad,
        max_aniso,
        padding_mode,
        interpolation_mode,
        align_corners,
        force_max_ansio,
        clip_grad);
    auto out = mipmap_aniso_grid_sampler_2d_cuda(
        input,
        grid,
        vt_dxdy_img,
        max_aniso,
        padding_mode,
        interpolation_mode,
        align_corners,
        force_max_ansio,
        clip_grad);
    return {out};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    int64_t mipmaps;
    bool requires_grad;
    int64_t max_aniso;
    int64_t padding_mode;
    int64_t interpolation_mode;
    bool align_corners;
    bool force_max_ansio;
    bool clip_grad;
    std::tie(
        mipmaps,
        requires_grad,
        max_aniso,
        padding_mode,
        interpolation_mode,
        align_corners,
        force_max_ansio,
        clip_grad) =
        ctx->saved_data["data"]
            .to<std::tuple<int64_t, bool, int64_t, int64_t, int64_t, bool, bool, bool>>();
    torch::autograd::tensor_list out;
    if (!requires_grad) {
      out.resize(mipmaps + 2);
      return out;
    }
    const auto saved = ctx->get_saved_variables();
    std::vector<torch::Tensor> input(saved.begin(), saved.begin() + mipmaps);
    torch::Tensor grid = saved[mipmaps];
    torch::Tensor vt_dxdy_img = saved[mipmaps + 1];
    auto grad_out = mipmap_aniso_grid_sampler_2d_cuda_backward(
        grad_outputs[0],
        input,
        grid,
        vt_dxdy_img,
        max_aniso,
        padding_mode,
        interpolation_mode,
        align_corners,
        force_max_ansio,
        clip_grad);
    std::vector<torch::Tensor> grads;

    grads.push_back(std::get<1>(grad_out));
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());

    for (auto& g : std::get<0>(grad_out)) {
      grads.push_back(g);
    }

    while (grads.size() < 19) {
      grads.push_back(torch::Tensor());
    }

    return grads;
  }
};

torch::Tensor mipmap_grid_sampler_2d_autograd(
    const torch::TensorList& input,
    const torch::Tensor& grid,
    const torch::Tensor& vt_dxdy_img,
    int64_t max_aniso,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool force_max_ansio,
    bool clip_grad) {
  return MipmapGridSample2DFunction::apply(
      grid,
      vt_dxdy_img,
      max_aniso,
      padding_mode,
      interpolation_mode,
      align_corners,
      force_max_ansio,
      clip_grad,
      input[0],
      input.size() > 1 ? input[1] : c10::optional<torch::Tensor>(),
      input.size() > 2 ? input[2] : c10::optional<torch::Tensor>(),
      input.size() > 3 ? input[3] : c10::optional<torch::Tensor>(),
      input.size() > 4 ? input[4] : c10::optional<torch::Tensor>(),
      input.size() > 5 ? input[5] : c10::optional<torch::Tensor>(),
      input.size() > 6 ? input[6] : c10::optional<torch::Tensor>(),
      input.size() > 7 ? input[7] : c10::optional<torch::Tensor>(),
      input.size() > 8 ? input[8] : c10::optional<torch::Tensor>(),
      input.size() > 9 ? input[9] : c10::optional<torch::Tensor>(),
      input.size() > 10 ? input[10] : c10::optional<torch::Tensor>())[0];
}

torch::Tensor mipmap_grid_sampler_2d_autocast(
    const torch::TensorList& input,
    const torch::Tensor& grid,
    const torch::Tensor& vt_dxdy_img,
    int64_t max_aniso,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool force_max_ansio,
    bool clip_grad) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return mipmap_grid_sampler_2d(
      at::autocast::cached_cast(torch::kFloat32, input),
      at::autocast::cached_cast(torch::kFloat32, grid),
      at::autocast::cached_cast(torch::kFloat32, vt_dxdy_img),
      max_aniso,
      padding_mode,
      interpolation_mode,
      align_corners,
      force_max_ansio,
      clip_grad);
}

#ifndef NO_PYBIND
// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
PYBIND11_MODULE(mipmap_grid_sampler_ext, m) {}
#endif

TORCH_LIBRARY(mipmap_grid_sampler_ext, m) {
  m.def(
      "mipmap_grid_sampler_2d(Tensor[] x, Tensor grid, Tensor vt_dxdy_img, int max_aniso, int padding_mode, int interpolation_mode, bool align_corners, bool force_max_ansio, bool clip_grad) -> Tensor");
}

TORCH_LIBRARY_IMPL(mipmap_grid_sampler_ext, Autograd, m) {
  m.impl("mipmap_grid_sampler_2d", &mipmap_grid_sampler_2d_autograd);
}

TORCH_LIBRARY_IMPL(mipmap_grid_sampler_ext, Autocast, m) {
  m.impl("mipmap_grid_sampler_2d", mipmap_grid_sampler_2d_autocast);
}

TORCH_LIBRARY_IMPL(mipmap_grid_sampler_ext, CUDA, m) {
  m.impl("mipmap_grid_sampler_2d", &mipmap_aniso_grid_sampler_2d_cuda);
}

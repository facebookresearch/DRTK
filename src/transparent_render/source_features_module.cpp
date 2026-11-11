// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the MIT license in the root LICENSE file.

#include <torch/script.h>
#include "source_features_kernel.h"

namespace {

using torch::autograd::tensor_list;

class SourceStencilFunction : public torch::autograd::Function<SourceStencilFunction> {
 public:
  static tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& v,
      const torch::Tensor& vi,
      const torch::Tensor& query_faces,
      const torch::Tensor& query_bary,
      const torch::Tensor& opacity,
      int64_t height,
      int64_t width,
      bool edge_gradients) {
    auto out = source_stencil_forward(v, vi, query_faces, query_bary, opacity, height, width);
    tensor_list saved{v, vi, query_faces, query_bary, opacity};
    saved.insert(saved.end(), out.begin(), out.end());
    ctx->save_for_backward(saved);
    ctx->saved_data["height"] = height;
    ctx->saved_data["width"] = width;
    ctx->saved_data["edge_gradients"] = edge_gradients && v.requires_grad();
    ctx->mark_non_differentiable({out[0]});
    ctx->set_materialize_grads(false);
    out.resize(4);
    return out;
  }

  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list gradients) {
    TORCH_CHECK(
        !torch::GradMode::is_enabled(), "source_transmittance supports first derivatives only");
    const auto saved = ctx->get_saved_variables();
    tensor_list grads;
    for (size_t i = 1; i < 4; ++i) {
      grads.push_back(
          gradients[i].defined() ? gradients[i].contiguous() : torch::zeros_like(saved[5 + i]));
    }
    auto out = source_stencil_backward(
        saved,
        grads,
        ctx->saved_data["height"].toInt(),
        ctx->saved_data["width"].toInt(),
        ctx->saved_data["edge_gradients"].toBool());
    return {out[0], {}, {}, out[1], out[2], {}, {}, {}};
  }
};

tensor_list stencil_cuda(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& opacity,
    int64_t h,
    int64_t w,
    bool /*edge_gradients*/) {
  auto out = source_stencil_forward(v, vi, qf, qb, opacity, h, w);
  out.resize(4);
  return out;
}

tensor_list stencil_autograd(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& opacity,
    int64_t h,
    int64_t w,
    bool edge_gradients) {
  if (torch::GradMode::is_enabled() &&
      (v.requires_grad() || qb.requires_grad() || opacity.requires_grad())) {
    return SourceStencilFunction::apply(v, vi, qf, qb, opacity, h, w, edge_gradients);
  }
  return stencil_cuda(v, vi, qf, qb, opacity, h, w, edge_gradients);
}

SourceShadingOptions
shading_options(bool mlp, double slope, int64_t mode, int64_t padding, int64_t activation) {
  TORCH_CHECK(
      mode >= 0 && mode < 3 && padding >= 0 && padding < 3 && activation >= 0 && activation < 4,
      "source transmittance: invalid shading options");
  return {
      mlp,
      static_cast<float>(slope),
      static_cast<int32_t>(mode),
      static_cast<int32_t>(padding),
      static_cast<int32_t>(activation)};
}

class SourceAppearanceFunction : public torch::autograd::Function<SourceAppearanceFunction> {
 public:
  static tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& v,
      const torch::Tensor& vi,
      const torch::Tensor& qf,
      const torch::Tensor& qb,
      const torch::Tensor& opacity,
      const torch::TensorList& appearance,
      int64_t height,
      int64_t width,
      bool mlp,
      double slope,
      int64_t mode,
      int64_t padding,
      int64_t activation,
      bool edges) {
    auto out = source_stencil_appearance_forward(
        v,
        vi,
        qf,
        qb,
        opacity,
        appearance,
        height,
        width,
        shading_options(mlp, slope, mode, padding, activation));
    tensor_list saved{v, vi, qf, qb, opacity};
    saved.insert(saved.end(), out.begin(), out.end());
    saved.insert(saved.end(), appearance.begin(), appearance.end());
    ctx->save_for_backward(saved);
    ctx->saved_data["height"] = height;
    ctx->saved_data["width"] = width;
    ctx->saved_data["mlp"] = mlp;
    ctx->saved_data["slope"] = slope;
    ctx->saved_data["mode"] = mode;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["activation"] = activation;
    ctx->saved_data["edges"] = edges && v.requires_grad();
    ctx->mark_non_differentiable({out[0]});
    ctx->set_materialize_grads(false);
    out.resize(4);
    return out;
  }

  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list gradients) {
    TORCH_CHECK(
        !torch::GradMode::is_enabled(), "source_transmittance supports first derivatives only");
    const auto saved = ctx->get_saved_variables();
    const torch::TensorList s(saved);
    tensor_list grads;
    for (size_t i = 1; i < 4; ++i) {
      grads.push_back(
          gradients[i].defined() ? gradients[i].contiguous() : torch::zeros_like(s[5 + i]));
    }
    const auto options = shading_options(
        ctx->saved_data["mlp"].toBool(),
        ctx->saved_data["slope"].toDouble(),
        ctx->saved_data["mode"].toInt(),
        ctx->saved_data["padding"].toInt(),
        ctx->saved_data["activation"].toInt());
    auto out = source_stencil_appearance_backward(
        s.slice(0, 16),
        grads,
        s.slice(16),
        ctx->saved_data["height"].toInt(),
        ctx->saved_data["width"].toInt(),
        ctx->saved_data["edges"].toBool(),
        options);
    tensor_list result{out[0], {}, {}, out[1], out[2]};
    result.insert(result.end(), out.begin() + 3, out.end());
    result.resize(result.size() + 8);
    return result;
  }
};

tensor_list appearance_cuda(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& opacity,
    const torch::TensorList& appearance,
    int64_t height,
    int64_t width,
    bool mlp,
    double slope,
    int64_t mode,
    int64_t padding,
    int64_t activation,
    bool /*edges*/) {
  auto out = source_stencil_appearance_forward(
      v,
      vi,
      qf,
      qb,
      opacity,
      appearance,
      height,
      width,
      shading_options(mlp, slope, mode, padding, activation));
  out.resize(4);
  return out;
}

tensor_list appearance_autograd(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& opacity,
    const torch::TensorList& appearance,
    int64_t height,
    int64_t width,
    bool mlp,
    double slope,
    int64_t mode,
    int64_t padding,
    int64_t activation,
    bool edges) {
  bool needs_grad = v.requires_grad() || qb.requires_grad() || opacity.requires_grad();
  for (const auto& t : appearance) {
    needs_grad |= t.requires_grad();
  }
  if (torch::GradMode::is_enabled() && needs_grad) {
    return SourceAppearanceFunction::apply(
        v,
        vi,
        qf,
        qb,
        opacity,
        appearance,
        height,
        width,
        mlp,
        slope,
        mode,
        padding,
        activation,
        edges);
  }
  return appearance_cuda(
      v,
      vi,
      qf,
      qb,
      opacity,
      appearance,
      height,
      width,
      mlp,
      slope,
      mode,
      padding,
      activation,
      edges);
}

class SampleStencilFunction : public torch::autograd::Function<SampleStencilFunction> {
 public:
  static tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& f,
      const torch::Tensor& indices,
      const torch::Tensor& weights) {
    ctx->save_for_backward({f, indices, weights});
    ctx->set_materialize_grads(false);
    return source_sample_forward(f, indices, weights);
  }

  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list grads) {
    TORCH_CHECK(
        !torch::GradMode::is_enabled(),
        "sample_with_transmittance supports first derivatives only");
    const auto saved = ctx->get_saved_variables();
    const auto& f = saved[0];
    const auto& indices = saved[1];
    const auto& weights = saved[2];
    const int64_t n = indices.size(0), q = indices.size(1);
    auto gs =
        grads[0].defined() ? grads[0].contiguous() : torch::zeros({n, f.size(1), q}, f.options());
    auto gc =
        grads[1].defined() ? grads[1].contiguous() : torch::zeros({n, 1, q}, weights.options());
    auto out = source_sample_backward(f, indices, weights, gs, gc);
    return {out[0], {}, out[1]};
  }
};

tensor_list sample_autograd(
    const torch::Tensor& f,
    const torch::Tensor& indices,
    const torch::Tensor& weights) {
  if (torch::GradMode::is_enabled() && (f.requires_grad() || weights.requires_grad())) {
    return SampleStencilFunction::apply(f, indices, weights);
  }
  return source_sample_forward(f, indices, weights);
}

} // namespace

TORCH_LIBRARY(source_features_ext, m) {
  m.def(
      "stencil(Tensor v, Tensor vi, Tensor query_faces, Tensor query_bary, Tensor opacity, "
      "int height, int width, bool edge_gradients=True) -> Tensor[]");
  m.def(
      "stencil_appearance(Tensor v, Tensor vi, Tensor query_faces, Tensor query_bary, Tensor opacity, "
      "Tensor[] appearance, int height, int width, bool mlp, float slope, int mode, int padding, "
      "int activation, bool edge_gradients=True) -> Tensor[]");
  m.def("sample(Tensor features, Tensor indices, Tensor weights) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(source_features_ext, Autograd, m) {
  m.impl("stencil", &stencil_autograd);
  m.impl("stencil_appearance", &appearance_autograd);
  m.impl("sample", &sample_autograd);
}

TORCH_LIBRARY_IMPL(source_features_ext, CUDA, m) {
  m.impl("stencil", &stencil_cuda);
  m.impl("stencil_appearance", &appearance_cuda);
  m.impl("sample", &source_sample_forward);
}

TORCH_LIBRARY_IMPL(source_features_ext, Autocast, m) {
  m.impl("stencil", torch::CppFunction::makeFallthrough());
  m.impl("stencil_appearance", torch::CppFunction::makeFallthrough());
  m.impl("sample", torch::CppFunction::makeFallthrough());
}

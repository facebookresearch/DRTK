// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/autocast_mode.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include <cmath>
#include <limits>
#include <vector>

#include "transparent_render_kernel.h"

namespace {

using torch::autograd::tensor_list;

TransparentDepthOptions depth_options(int64_t mode, double background, double alpha_threshold) {
  TORCH_CHECK(mode >= 0 && mode <= 3, "transparent render: invalid depth mode");
  TORCH_CHECK(
      std::isfinite(static_cast<float>(background)) && background > 0.0,
      "transparent render: background depth must be positive and finite in float32");
  TORCH_CHECK(
      std::isfinite(static_cast<float>(alpha_threshold)) &&
          static_cast<float>(alpha_threshold) >= std::numeric_limits<float>::min() &&
          alpha_threshold < 1.0,
      "transparent render: depth alpha threshold must be positive and less than one in float32");
  return {
      static_cast<TransparentDepthMode>(mode),
      static_cast<float>(background),
      static_cast<float>(alpha_threshold)};
}

TransparentDepthOptions saved_depth_options(torch::autograd::AutogradContext* ctx) {
  return depth_options(
      ctx->saved_data["depth_mode"].toInt(),
      ctx->saved_data["background_depth"].toDouble(),
      ctx->saved_data["depth_alpha_threshold"].toDouble());
}

tensor_list render_with_texture(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& texture,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    bool edge_gradients,
    int64_t depth_mode,
    double background_depth,
    double depth_alpha_threshold) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("transparent_render_ext::render_with_texture", "")
                       .typed<decltype(render_with_texture)>();
  return op.call(
      v,
      vi,
      uv,
      texture,
      opacity,
      background,
      height,
      width,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation,
      edge_gradients,
      depth_mode,
      background_depth,
      depth_alpha_threshold);
}

tensor_list render_with_mlp(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& reference_uv,
    const torch::Tensor& texture,
    const torch::Tensor& reference_texture,
    const torch::TensorList& weights,
    const torch::TensorList& biases,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    double negative_slope,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    bool edge_gradients,
    int64_t depth_mode,
    double background_depth,
    double depth_alpha_threshold) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("transparent_render_ext::render_with_mlp", "")
                       .typed<decltype(render_with_mlp)>();
  return op.call(
      v,
      vi,
      uv,
      reference_uv,
      texture,
      reference_texture,
      weights,
      biases,
      opacity,
      background,
      height,
      width,
      negative_slope,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation,
      edge_gradients,
      depth_mode,
      background_depth,
      depth_alpha_threshold);
}

tensor_list render_with_texture_cuda(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& texture,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    bool edge_gradients,
    int64_t depth_mode,
    double background_depth,
    double depth_alpha_threshold) {
  auto outputs = transparent_render_with_texture_cuda_forward(
      v,
      vi,
      uv,
      texture,
      opacity,
      background,
      height,
      width,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation,
      depth_options(depth_mode, background_depth, depth_alpha_threshold),
      false);
  outputs.resize(3);
  return outputs;
}

tensor_list render_with_mlp_cuda(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& reference_uv,
    const torch::Tensor& texture,
    const torch::Tensor& reference_texture,
    const torch::TensorList& weights,
    const torch::TensorList& biases,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    double negative_slope,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    bool edge_gradients,
    int64_t depth_mode,
    double background_depth,
    double depth_alpha_threshold) {
  auto outputs = transparent_render_with_mlp_cuda_forward(
      v,
      vi,
      uv,
      reference_uv,
      texture,
      reference_texture,
      weights,
      biases,
      opacity,
      background,
      height,
      width,
      negative_slope,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation,
      depth_options(depth_mode, background_depth, depth_alpha_threshold),
      false);
  outputs.resize(3);
  return outputs;
}

tensor_list render_with_texture_cpu(
    const torch::Tensor& /*v*/,
    const torch::Tensor& /*vi*/,
    const torch::Tensor& /*uv*/,
    const torch::Tensor& /*texture*/,
    const torch::Tensor& /*opacity*/,
    const torch::Tensor& /*background*/,
    int64_t /*height*/,
    int64_t /*width*/,
    int64_t /*interpolation_mode*/,
    int64_t /*padding_mode*/,
    double /*transmittance_epsilon*/,
    int64_t /*rgb_activation*/,
    int64_t /*alpha_activation*/,
    bool /*edge_gradients*/,
    int64_t /*depth_mode*/,
    double /*background_depth*/,
    double /*depth_alpha_threshold*/) {
  TORCH_CHECK(
      false,
      "transparent render: render_with_texture requires CUDA; "
      "use render_transparent_ref for a CPU correctness oracle");
}

tensor_list render_with_mlp_cpu(
    const torch::Tensor& /*v*/,
    const torch::Tensor& /*vi*/,
    const torch::Tensor& /*uv*/,
    const torch::Tensor& /*reference_uv*/,
    const torch::Tensor& /*texture*/,
    const torch::Tensor& /*reference_texture*/,
    const torch::TensorList& /*weights*/,
    const torch::TensorList& /*biases*/,
    const torch::Tensor& /*opacity*/,
    const torch::Tensor& /*background*/,
    int64_t /*height*/,
    int64_t /*width*/,
    double /*negative_slope*/,
    int64_t /*interpolation_mode*/,
    int64_t /*padding_mode*/,
    double /*transmittance_epsilon*/,
    int64_t /*rgb_activation*/,
    int64_t /*alpha_activation*/,
    bool /*edge_gradients*/,
    int64_t /*depth_mode*/,
    double /*background_depth*/,
    double /*depth_alpha_threshold*/) {
  TORCH_CHECK(
      false,
      "transparent render: render_with_mlp requires CUDA; "
      "use render_transparent_mlp_ref for a CPU correctness oracle");
}

bool render_with_texture_requires_grad(
    const torch::Tensor& v,
    const torch::Tensor& uv,
    const torch::Tensor& texture,
    const torch::Tensor& opacity,
    const torch::Tensor& background) {
  return torch::GradMode::is_enabled() &&
      (v.requires_grad() || uv.requires_grad() || texture.requires_grad() ||
       opacity.requires_grad() || background.requires_grad());
}

bool render_with_mlp_requires_grad(
    const torch::Tensor& v,
    const torch::Tensor& uv,
    const torch::Tensor& reference_uv,
    const torch::Tensor& texture,
    const torch::Tensor& reference_texture,
    const torch::TensorList& weights,
    const torch::TensorList& biases,
    const torch::Tensor& opacity,
    const torch::Tensor& background) {
  if (!torch::GradMode::is_enabled()) {
    return false;
  }
  if (v.requires_grad() || uv.requires_grad() || reference_uv.requires_grad() ||
      texture.requires_grad() || reference_texture.requires_grad() || opacity.requires_grad() ||
      background.requires_grad()) {
    return true;
  }
  for (const auto& parameter : weights) {
    if (parameter.requires_grad()) {
      return true;
    }
  }
  for (const auto& parameter : biases) {
    if (parameter.requires_grad()) {
      return true;
    }
  }
  return false;
}

void check_backward_compositing_options(double transmittance_epsilon, const char* variant) {
  TORCH_CHECK(
      transmittance_epsilon == 0.0,
      "transparent render ",
      variant,
      ": transmittance_epsilon must be zero when gradients are enabled");
}

class RenderWithTextureFunction : public torch::autograd::Function<RenderWithTextureFunction> {
 public:
  static tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& v,
      const torch::Tensor& vi,
      const torch::Tensor& uv,
      const torch::Tensor& texture,
      const torch::Tensor& opacity,
      const torch::Tensor& background,
      int64_t height,
      int64_t width,
      int64_t interpolation_mode,
      int64_t padding_mode,
      double transmittance_epsilon,
      int64_t rgb_activation,
      int64_t alpha_activation,
      bool edge_gradients,
      int64_t depth_mode,
      double background_depth,
      double depth_alpha_threshold) {
    TORCH_CHECK(
        v.is_cuda(),
        "transparent render: render_with_texture requires CUDA; "
        "use render_transparent_ref for a CPU correctness oracle");
    auto outputs = transparent_render_with_texture_cuda_forward(
        v,
        vi,
        uv,
        texture,
        opacity,
        background,
        height,
        width,
        interpolation_mode,
        padding_mode,
        transmittance_epsilon,
        rgb_activation,
        alpha_activation,
        depth_options(depth_mode, background_depth, depth_alpha_threshold));
    ctx->save_for_backward(
        {v,
         vi,
         uv,
         texture,
         opacity,
         background,
         outputs[3],
         outputs[4],
         outputs[5],
         outputs[6],
         outputs[1],
         outputs[2],
         outputs[7],
         outputs[8]});
    ctx->saved_data["interpolation_mode"] = interpolation_mode;
    ctx->saved_data["padding_mode"] = padding_mode;
    ctx->saved_data["rgb_activation"] = rgb_activation;
    ctx->saved_data["alpha_activation"] = alpha_activation;
    ctx->saved_data["depth_mode"] = depth_mode;
    ctx->saved_data["background_depth"] = background_depth;
    ctx->saved_data["depth_alpha_threshold"] = depth_alpha_threshold;
    ctx->saved_data["edge_gradients"] = edge_gradients && v.requires_grad();
    return {outputs[0], outputs[1], outputs[2]};
  }

  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto grads = transparent_render_with_texture_cuda_backward(
        saved[0],
        saved[1],
        saved[2],
        saved[3],
        saved[4],
        saved[5],
        saved[6],
        saved[7],
        saved[8],
        saved[9],
        saved[13],
        grad_outputs[0].contiguous(),
        grad_outputs[1].contiguous(),
        ctx->saved_data["interpolation_mode"].toInt(),
        ctx->saved_data["padding_mode"].toInt(),
        ctx->saved_data["rgb_activation"].toInt(),
        ctx->saved_data["alpha_activation"].toInt(),
        saved[10],
        saved[11],
        saved[12],
        grad_outputs[2].contiguous(),
        saved_depth_options(ctx),
        ctx->saved_data["edge_gradients"].toBool() && ctx->needs_input_grad(0));
    return {
        grads[0],
        torch::Tensor(),
        grads[1],
        grads[2],
        grads[3],
        grads[4],
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()};
  }
};

class RenderWithMlpFunction : public torch::autograd::Function<RenderWithMlpFunction> {
 public:
  static tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& v,
      const torch::Tensor& vi,
      const torch::Tensor& uv,
      const torch::Tensor& reference_uv,
      const torch::Tensor& texture,
      const torch::Tensor& reference_texture,
      const torch::TensorList& weights,
      const torch::TensorList& biases,
      const torch::Tensor& opacity,
      const torch::Tensor& background,
      int64_t height,
      int64_t width,
      double negative_slope,
      int64_t interpolation_mode,
      int64_t padding_mode,
      double transmittance_epsilon,
      int64_t rgb_activation,
      int64_t alpha_activation,
      bool edge_gradients,
      int64_t depth_mode,
      double background_depth,
      double depth_alpha_threshold) {
    TORCH_CHECK(
        v.is_cuda(),
        "transparent render: render_with_mlp requires CUDA; "
        "use render_transparent_mlp_ref for a CPU correctness oracle");
    auto outputs = transparent_render_with_mlp_cuda_forward(
        v,
        vi,
        uv,
        reference_uv,
        texture,
        reference_texture,
        weights,
        biases,
        opacity,
        background,
        height,
        width,
        negative_slope,
        interpolation_mode,
        padding_mode,
        transmittance_epsilon,
        rgb_activation,
        alpha_activation,
        depth_options(depth_mode, background_depth, depth_alpha_threshold));

    std::vector<torch::Tensor> saved{v, vi, uv, reference_uv, texture, reference_texture};
    saved.reserve(16 + weights.size() + biases.size());
    saved.insert(saved.end(), weights.begin(), weights.end());
    saved.insert(saved.end(), biases.begin(), biases.end());
    saved.insert(
        saved.end(),
        {opacity,
         background,
         outputs[3],
         outputs[4],
         outputs[5],
         outputs[6],
         outputs[1],
         outputs[2],
         outputs[7],
         outputs[8]});
    ctx->save_for_backward(saved);
    ctx->saved_data["layer_count"] = static_cast<int64_t>(weights.size());
    ctx->saved_data["negative_slope"] = negative_slope;
    ctx->saved_data["interpolation_mode"] = interpolation_mode;
    ctx->saved_data["padding_mode"] = padding_mode;
    ctx->saved_data["rgb_activation"] = rgb_activation;
    ctx->saved_data["alpha_activation"] = alpha_activation;
    ctx->saved_data["depth_mode"] = depth_mode;
    ctx->saved_data["background_depth"] = background_depth;
    ctx->saved_data["depth_alpha_threshold"] = depth_alpha_threshold;
    ctx->saved_data["edge_gradients"] = edge_gradients && v.requires_grad();
    return {outputs[0], outputs[1], outputs[2]};
  }

  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const auto layer_count = ctx->saved_data["layer_count"].toInt();
    const torch::TensorList weights(saved.data() + 6, layer_count);
    const torch::TensorList biases(saved.data() + 6 + layer_count, layer_count);
    const auto opacity_index = 6 + 2 * layer_count;
    const auto background_index = opacity_index + 1;
    const auto tile_offsets_index = background_index + 1;
    const auto sorted_face_ids_index = tile_offsets_index + 1;
    const auto saved_transmittance_index = sorted_face_ids_index + 1;
    const auto saved_transmittance_exponent_index = saved_transmittance_index + 1;

    auto grads = transparent_render_with_mlp_cuda_backward(
        saved[0],
        saved[1],
        saved[2],
        saved[3],
        saved[4],
        saved[5],
        weights,
        biases,
        saved[opacity_index],
        saved[background_index],
        saved[tile_offsets_index],
        saved[sorted_face_ids_index],
        saved[saved_transmittance_index],
        saved[saved_transmittance_exponent_index],
        saved[saved_transmittance_exponent_index + 4],
        grad_outputs[0].contiguous(),
        grad_outputs[1].contiguous(),
        ctx->saved_data["negative_slope"].toDouble(),
        ctx->saved_data["interpolation_mode"].toInt(),
        ctx->saved_data["padding_mode"].toInt(),
        ctx->saved_data["rgb_activation"].toInt(),
        ctx->saved_data["alpha_activation"].toInt(),
        saved[saved_transmittance_exponent_index + 1],
        saved[saved_transmittance_exponent_index + 2],
        saved[saved_transmittance_exponent_index + 3],
        grad_outputs[2].contiguous(),
        saved_depth_options(ctx),
        ctx->saved_data["edge_gradients"].toBool() && ctx->needs_input_grad(0));

    tensor_list result{grads[0], torch::Tensor(), grads[1], grads[2], grads[3], grads[4]};
    result.reserve(20 + 2 * layer_count);
    result.insert(result.end(), grads.begin() + 7, grads.begin() + 7 + layer_count);
    result.insert(
        result.end(), grads.begin() + 7 + layer_count, grads.begin() + 7 + 2 * layer_count);
    result.push_back(grads[5]);
    result.push_back(grads[6]);
    result.insert(result.end(), 12, torch::Tensor());
    return result;
  }
};

tensor_list render_with_texture_autograd(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& texture,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    bool edge_gradients,
    int64_t depth_mode,
    double background_depth,
    double depth_alpha_threshold) {
  if (!render_with_texture_requires_grad(v, uv, texture, opacity, background)) {
    return render_with_texture_cuda(
        v,
        vi,
        uv,
        texture,
        opacity,
        background,
        height,
        width,
        interpolation_mode,
        padding_mode,
        transmittance_epsilon,
        rgb_activation,
        alpha_activation,
        edge_gradients,
        depth_mode,
        background_depth,
        depth_alpha_threshold);
  }
  check_backward_compositing_options(transmittance_epsilon, "render_with_texture");
  return RenderWithTextureFunction::apply(
      v,
      vi,
      uv,
      texture,
      opacity,
      background,
      height,
      width,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation,
      edge_gradients,
      depth_mode,
      background_depth,
      depth_alpha_threshold);
}

tensor_list render_with_mlp_autograd(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& reference_uv,
    const torch::Tensor& texture,
    const torch::Tensor& reference_texture,
    const torch::TensorList& weights,
    const torch::TensorList& biases,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    double negative_slope,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    bool edge_gradients,
    int64_t depth_mode,
    double background_depth,
    double depth_alpha_threshold) {
  if (!render_with_mlp_requires_grad(
          v, uv, reference_uv, texture, reference_texture, weights, biases, opacity, background)) {
    return render_with_mlp_cuda(
        v,
        vi,
        uv,
        reference_uv,
        texture,
        reference_texture,
        weights,
        biases,
        opacity,
        background,
        height,
        width,
        negative_slope,
        interpolation_mode,
        padding_mode,
        transmittance_epsilon,
        rgb_activation,
        alpha_activation,
        edge_gradients,
        depth_mode,
        background_depth,
        depth_alpha_threshold);
  }
  check_backward_compositing_options(transmittance_epsilon, "render_with_mlp");
  return RenderWithMlpFunction::apply(
      v,
      vi,
      uv,
      reference_uv,
      texture,
      reference_texture,
      weights,
      biases,
      opacity,
      background,
      height,
      width,
      negative_slope,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation,
      edge_gradients,
      depth_mode,
      background_depth,
      depth_alpha_threshold);
}

tensor_list render_with_texture_autocast(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& texture,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    bool edge_gradients,
    int64_t depth_mode,
    double background_depth,
    double depth_alpha_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return render_with_texture(
      at::autocast::cached_cast(torch::kFloat32, v),
      vi,
      at::autocast::cached_cast(torch::kFloat32, uv),
      at::autocast::cached_cast(torch::kFloat32, texture),
      at::autocast::cached_cast(torch::kFloat32, opacity),
      at::autocast::cached_cast(torch::kFloat32, background),
      height,
      width,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation,
      edge_gradients,
      depth_mode,
      background_depth,
      depth_alpha_threshold);
}

tensor_list render_with_mlp_autocast(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& uv,
    const torch::Tensor& reference_uv,
    const torch::Tensor& texture,
    const torch::Tensor& reference_texture,
    const torch::TensorList& weights,
    const torch::TensorList& biases,
    const torch::Tensor& opacity,
    const torch::Tensor& background,
    int64_t height,
    int64_t width,
    double negative_slope,
    int64_t interpolation_mode,
    int64_t padding_mode,
    double transmittance_epsilon,
    int64_t rgb_activation,
    int64_t alpha_activation,
    bool edge_gradients,
    int64_t depth_mode,
    double background_depth,
    double depth_alpha_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  std::vector<torch::Tensor> cast_weights;
  std::vector<torch::Tensor> cast_biases;
  cast_weights.reserve(weights.size());
  cast_biases.reserve(biases.size());
  for (const auto& weight : weights) {
    cast_weights.push_back(at::autocast::cached_cast(torch::kFloat32, weight));
  }
  for (const auto& bias : biases) {
    cast_biases.push_back(at::autocast::cached_cast(torch::kFloat32, bias));
  }
  return render_with_mlp(
      at::autocast::cached_cast(torch::kFloat32, v),
      vi,
      at::autocast::cached_cast(torch::kFloat32, uv),
      at::autocast::cached_cast(torch::kFloat32, reference_uv),
      at::autocast::cached_cast(torch::kFloat32, texture),
      at::autocast::cached_cast(torch::kFloat32, reference_texture),
      cast_weights,
      cast_biases,
      at::autocast::cached_cast(torch::kFloat32, opacity),
      at::autocast::cached_cast(torch::kFloat32, background),
      height,
      width,
      negative_slope,
      interpolation_mode,
      padding_mode,
      transmittance_epsilon,
      rgb_activation,
      alpha_activation,
      edge_gradients,
      depth_mode,
      background_depth,
      depth_alpha_threshold);
}

} // namespace

#ifndef NO_PYBIND
PYBIND11_MODULE(transparent_render_ext, m) {}
#endif

TORCH_LIBRARY(transparent_render_ext, m) {
  m.def(
      "render_with_texture(Tensor v, Tensor vi, Tensor uv, Tensor texture, Tensor opacity, "
      "Tensor background, int height, int width, int interpolation_mode, int padding_mode, "
      "float transmittance_epsilon, int rgb_activation, int alpha_activation, bool edge_gradients=True, "
      "int depth_mode=0, float background_depth=1000., float depth_alpha_threshold=1.e-6) -> Tensor[]");
  m.def(
      "render_with_mlp(Tensor v, Tensor vi, Tensor uv0, Tensor uv1, Tensor texture0, "
      "Tensor texture1, Tensor[] weights, Tensor[] biases, Tensor opacity, "
      "Tensor background, int height, int width, float negative_slope, "
      "int interpolation_mode, int padding_mode, float transmittance_epsilon, "
      "int rgb_activation, int alpha_activation, bool edge_gradients=True, "
      "int depth_mode=0, float background_depth=1000., float depth_alpha_threshold=1.e-6) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(transparent_render_ext, Autograd, m) {
  m.impl("render_with_texture", &render_with_texture_autograd);
  m.impl("render_with_mlp", &render_with_mlp_autograd);
}

TORCH_LIBRARY_IMPL(transparent_render_ext, Autocast, m) {
  m.impl("render_with_texture", &render_with_texture_autocast);
  m.impl("render_with_mlp", &render_with_mlp_autocast);
}

TORCH_LIBRARY_IMPL(transparent_render_ext, CUDA, m) {
  m.impl("render_with_texture", &render_with_texture_cuda);
  m.impl("render_with_mlp", &render_with_mlp_cuda);
}

TORCH_LIBRARY_IMPL(transparent_render_ext, CPU, m) {
  m.impl("render_with_texture", &render_with_texture_cpu);
  m.impl("render_with_mlp", &render_with_mlp_cpu);
}

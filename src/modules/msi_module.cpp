#include <torch/script.h>
#include "../msi_bkg/msi_bkg.h"

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

/*
 * Renders a Multi-Sphere Image which is similar to the one described in "NeRF++: Analyzing and
 * Improving Neural Radiance Fields".
 * This file provides python/torch bindings and autograd function implementation.
 * It main implementation is in `src/msi_bkg/msi_bkg.cu`, see functions `msi_bkg_forward_cuda` and
 * `msi_bkg_backward_cuda`.
 * For more details see docstring in drtk/renderlayer/msi.py
 */

class MSIFunction : public torch::autograd::Function<MSIFunction> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& ray_o,
      const torch::Tensor& ray_d,
      const torch::Tensor& texture,
      int64_t sub_step_count,
      double min_inv_r,
      double max_inv_r,
      double stop_thresh) {
    ctx->set_materialize_grads(false);
    std::vector<torch::Tensor> save_list;

    save_list.push_back(ray_o);
    save_list.push_back(ray_d);
    save_list.push_back(texture);

    bool requires_grad = texture.requires_grad();

    ctx->saved_data["data"] =
        std::make_tuple(requires_grad, sub_step_count, min_inv_r, max_inv_r, stop_thresh);
    torch::Tensor rgba_img = msi_bkg_forward_cuda(
        ray_o, ray_d, texture, sub_step_count, min_inv_r, max_inv_r, stop_thresh);

    save_list.push_back(rgba_img);
    ctx->save_for_backward(save_list);

    return rgba_img;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      // rgba_img
      torch::autograd::tensor_list grad_outputs) {
    bool requires_grad;
    int64_t sub_step_count;
    double stop_thresh;
    double min_inv_r;
    double max_inv_r;

    std::tie(requires_grad, sub_step_count, min_inv_r, max_inv_r, stop_thresh) =
        ctx->saved_data["data"].to<std::tuple<bool, int64_t, double, double, double>>();
    torch::autograd::tensor_list grads;
    if (!requires_grad) {
      grads.resize(7); // 7 - number of arguments of the forward function, see comment below.
      return grads;
    }

    const auto saved = ctx->get_saved_variables();
    auto ray_o = saved[0];
    auto ray_d = saved[1];
    auto texture = saved[2];
    auto rgba_img = saved[3];

    auto rgba_img_grad = grad_outputs[0];

    auto texture_grad = msi_bkg_backward_cuda(
        rgba_img,
        rgba_img_grad,
        ray_o,
        ray_d,
        texture,
        sub_step_count,
        min_inv_r,
        max_inv_r,
        stop_thresh);

    // The output has to be a vector of tensors, the legth of wich must match the number of
    // arguments in the forward function. Even if the arhument is not a tensor and can not have a
    // gradients.
    // We do not compute gradints with respect to ray origin
    // or direction, other inputs except `texture` are not tensors (can't have gradints).
    // Thus we only provide gradient for the `texture` which is the third argument.
    auto output_has_no_grad = torch::Tensor();
    grads.push_back(output_has_no_grad);
    grads.push_back(output_has_no_grad);
    grads.push_back(texture_grad);
    grads.push_back(output_has_no_grad);
    grads.push_back(output_has_no_grad);
    grads.push_back(output_has_no_grad);
    grads.push_back(output_has_no_grad);
    return grads;
  }
};

torch::Tensor msi_bkg_autograd(
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& texture,
    int64_t sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh) {
  return MSIFunction::apply(
      ray_o, ray_d, texture, sub_step_count, min_inv_r, max_inv_r, stop_thresh);
}

#ifndef NO_PYBIND
PYBIND11_MODULE(msi_ext, m) {
  m.def("msi_bkg", &msi_bkg_autograd);
}
#endif

TORCH_LIBRARY(msi_ext, m) {
  m.def(
      "msi_bkg(Tensor ray_o, Tensor ray_d, Tensor texture, "
      "int sub_step_count, float min_inv_r, float max_inv_r, float stop_thresh) -> "
      "Tensor");
}

TORCH_LIBRARY_IMPL(msi_ext, Autograd, m) {
  m.impl("msi_bkg", &msi_bkg_autograd);
}

TORCH_LIBRARY_IMPL(msi_ext, CUDA, m) {
  m.impl("msi_bkg", &msi_bkg_forward_cuda);
}

#include <torch/script.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "interpolate_kernel.h"

class ComputeVertImageFunction : public torch::autograd::Function<ComputeVertImageFunction> {
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
    return {compute_vert_image_cuda(vert_attributes, vi, index_img, bary_img)};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    torch::Tensor vert_attributes = saved[0];
    torch::Tensor vi = saved[1];
    torch::Tensor index_img = saved[2];
    torch::Tensor bary_img = saved[3];
    bool bary_img_requires_grad = bary_img.requires_grad();
    bool vert_requires_grad = vert_attributes.requires_grad();

    torch::autograd::tensor_list out;
    if (!bary_img_requires_grad && !vert_requires_grad) {
      out.resize(4);
      return out;
    }
    auto grad_out =
        compute_vert_image_cuda_backward(grad_outputs[0], vert_attributes, vi, index_img, bary_img);

    out.push_back(std::get<0>(grad_out));
    out.push_back(torch::Tensor());
    out.push_back(torch::Tensor());
    out.push_back(std::get<1>(grad_out));
    return out;
  }
};

torch::Tensor compute_vert_image_autograd(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  return ComputeVertImageFunction::apply(vert_attributes, vi, index_img, bary_img)[0];
}

#ifndef NO_PYBIND
// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
PYBIND11_MODULE(interpolate_ext, m) {
  m.def("compute_vert_image", &compute_vert_image_cuda);
}
#endif

TORCH_LIBRARY(interpolate_ext, m) {
  m.def(
      "compute_vert_image(Tensor vert_attributes, Tensor vi, Tensor index_img, Tensor bary_img) -> Tensor");
}

TORCH_LIBRARY_IMPL(interpolate_ext, Autograd, m) {
  m.impl("compute_vert_image", &compute_vert_image_autograd);
}

TORCH_LIBRARY_IMPL(interpolate_ext, CUDA, m) {
  m.impl("compute_vert_image", &compute_vert_image_cuda);
}

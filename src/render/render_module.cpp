#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "../include/common.h"
#include "render_kernel.h"

std::vector<torch::Tensor> render_forward(
    torch::Tensor v2d,
    torch::Tensor vt,
    at::optional<torch::Tensor> vn,
    torch::Tensor vi,
    torch::Tensor vti,
    torch::Tensor indeximg,
    torch::Tensor depthimg,
    torch::Tensor baryimg,
    torch::Tensor uvimg,
    at::optional<torch::Tensor> vnimg) {
  CHECK_INPUT(v2d);
  CHECK_INPUT(vt);
  CHECK_INPUT(vi);
  CHECK_INPUT(vti);
  CHECK_INPUT(indeximg);
  CHECK_INPUT(depthimg);
  CHECK_INPUT(baryimg);
  CHECK_INPUT(uvimg);

  int N = indeximg.size(0);
  int H = indeximg.size(1);
  int W = indeximg.size(2);
  int V = v2d.size(1);

  if (vn) {
    MYCHECK(vnimg, "Provided vn but missing input vn_img tensor.");
    CHECK_INPUT((*vn));
    CHECK_INPUT((*vnimg));
    MYCHECK(
        (vn->sizes().size() == 3 && vn->sizes()[2] == 3),
        "Expected vn with shape [N, V, 3], got ",
        vn->sizes());
    MYCHECK(
        (vnimg->sizes().size() == 4 && vnimg->sizes()[3] == 3),
        "Expected vn_img with shape [N, H, W, 3], got ",
        vnimg->sizes());
  }

  at::cuda::CUDAGuard device_guard(v2d.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  render_forward_cuda(
      N,
      H,
      W,
      V,
      DATA_PTR(v2d, float),
      DATA_PTR(vt, float),
      vn ? DATA_PTR((*vn), float) : nullptr,
      DATA_PTR(vi, int32_t),
      DATA_PTR(vti, int32_t),
      DATA_PTR(indeximg, int32_t),
      DATA_PTR(depthimg, float),
      DATA_PTR(baryimg, float),
      DATA_PTR(uvimg, float),
      vnimg ? DATA_PTR((*vnimg), float) : nullptr,
      stream);

  return {};
}

std::vector<torch::Tensor> render_backward(
    torch::Tensor v2d,
    torch::Tensor vt,
    at::optional<torch::Tensor> vn,
    torch::Tensor vi,
    torch::Tensor vti,
    torch::Tensor indeximg,
    torch::Tensor grad_depthimg,
    torch::Tensor grad_baryimg,
    torch::Tensor grad_uvimg,
    at::optional<torch::Tensor> grad_vnimg,
    torch::Tensor grad_v2d,
    at::optional<torch::Tensor> grad_vn) {
  CHECK_INPUT(v2d);
  CHECK_INPUT(vt);
  CHECK_INPUT(vi);
  CHECK_INPUT(vti);
  CHECK_INPUT(indeximg);
  CHECK_INPUT(grad_depthimg);
  CHECK_INPUT(grad_baryimg);
  CHECK_INPUT(grad_uvimg);
  CHECK_INPUT(grad_v2d);

  int N = indeximg.size(0);
  int H = indeximg.size(1);
  int W = indeximg.size(2);
  int V = v2d.size(1);

  if (vn) {
    MYCHECK(grad_vnimg, "Provided vn but missing output grad for vn_img.");
    MYCHECK(grad_vn, "Provided vn but missing input grad_vn tensor.");
    CHECK_INPUT((*vn));
    CHECK_INPUT((*grad_vnimg));
    CHECK_INPUT((*grad_vn));
    MYCHECK(
        (vn->sizes().size() == 3 && vn->sizes()[2] == 3),
        "Expected vn with shape [N, V, 3], got ",
        vn->sizes());
    MYCHECK(
        (grad_vn->sizes().size() == 3 && grad_vn->sizes()[2] == 3),
        "Expected grad_vn with shape [N, V, 3], got ",
        grad_vn->sizes());
    MYCHECK(
        (grad_vnimg->sizes().size() == 4 && grad_vnimg->sizes()[3] == 3),
        "Expected grad_vnimg with shape [N, H, W, 3], got ",
        grad_vnimg->sizes());
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  render_backward_cuda(
      N,
      H,
      W,
      V,
      DATA_PTR(v2d, float),
      DATA_PTR(vt, float),
      vn ? DATA_PTR((*vn), float) : nullptr,
      DATA_PTR(vi, int32_t),
      DATA_PTR(vti, int32_t),
      DATA_PTR(indeximg, int32_t),
      DATA_PTR(grad_depthimg, float),
      DATA_PTR(grad_baryimg, float),
      DATA_PTR(grad_uvimg, float),
      grad_vnimg ? DATA_PTR((*grad_vnimg), float) : nullptr,
      DATA_PTR(grad_v2d, float),
      grad_vn ? DATA_PTR((*grad_vn), float) : nullptr,
      stream);

  return {};
}

#ifndef NO_PYBIND
PYBIND11_MODULE(render_ext, m) {}
#endif

TORCH_LIBRARY(render_ext, m) {
  m.def("render_forward", &render_forward);
  m.def("render_backward", &render_backward);
}

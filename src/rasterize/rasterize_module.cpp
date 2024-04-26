#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>

#include "../include/common.h"
#include "../rasterize/rasterize_kernel.h"

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

int64_t rasterize_packed(
    torch::Tensor verts_t,
    torch::Tensor vind_t,
    torch::Tensor depth_img_t,
    torch::Tensor index_img_t,
    torch::Tensor packedindex_img_t) {
  CHECK_INPUT(verts_t)
  CHECK_INPUT(vind_t)
  CHECK_INPUT(depth_img_t)
  CHECK_INPUT(index_img_t)
  CHECK_INPUT(packedindex_img_t)
  CHECK_3DIMS(index_img_t)
  CHECK_3DIMS(verts_t)
  CHECK_3DIMS(depth_img_t)

  at::cuda::CUDAGuard device_guard(verts_t.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int nprims = vind_t.sizes()[0];
  const int nverts = verts_t.sizes()[1];
  const int b = verts_t.sizes()[0];
  const int h = index_img_t.sizes()[1];
  const int w = index_img_t.sizes()[2];
  TORCH_CHECK(index_img_t.numel() >= w * h, "Bad index image shape.");
  TORCH_CHECK(depth_img_t.numel() >= w * h, "Bad depth image shape.");
  TORCH_CHECK(vind_t.sizes()[1] == 3, "Vertex indices must have shape[1] == 3.");
  TORCH_CHECK(verts_t.sizes()[2] == 3, "Vertices must be 3D points, shape[2] == 3.");
  TORCH_CHECK(packedindex_img_t.sizes()[0] == b, "Packed index image has wrong batch size.");
  TORCH_CHECK(depth_img_t.sizes()[0] == b, "Depth image has wrong batch size.");
  TORCH_CHECK(index_img_t.sizes()[0] == b, "Index image has wrong batch size.");

  int3* vind = reinterpret_cast<int3*>(DATA_PTR(vind_t, int));
  float* verts = DATA_PTR(verts_t, float);
  float* depth_img = DATA_PTR(depth_img_t, float);
  int* index_img = DATA_PTR(index_img_t, int);
  int* packedindex_img = DATA_PTR(packedindex_img_t, int);

  packed_rasterize_cuda(
      b,
      w,
      h,
      nprims,
      nverts,
      reinterpret_cast<float3*>(verts),
      vind,
      depth_img,
      index_img,
      packedindex_img,
      stream);

  return 1;
}

#ifndef NO_PYBIND
PYBIND11_MODULE(rasterizer_ext, m) {
  m.def(
      "rasterize_packed",
      &rasterize_packed,
      py::call_guard<py::gil_scoped_release>(),
      "Rasterize with packed depth / index buffer.");
}
#endif

TORCH_LIBRARY(rasterizer_ext, m) {
  m.def("rasterize_packed", rasterize_packed);
}

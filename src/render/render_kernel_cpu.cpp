// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Parallel.h>
#include <cpu_atomic.h>
#include <cuda_math_helper.h>
#include <torch/types.h>

#include "render_kernel.h"

using namespace math;
using drtk::atomic_add;

namespace {

template <typename scalar_t>
void render_forward_cpu_impl(
    const scalar_t* v_ptr,
    const int32_t* vi_ptr,
    const int32_t* index_img_ptr,
    scalar_t* depth_img_ptr,
    scalar_t* bary_img_ptr,
    int64_t N,
    int64_t V,
    int64_t H,
    int64_t W,
    // v strides
    int64_t v_sN,
    int64_t v_sV,
    int64_t v_sC,
    // vi strides
    int64_t vi_sN,
    int64_t vi_sV,
    int64_t vi_sF,
    // index_img strides
    int64_t index_img_sN,
    int64_t index_img_sH,
    int64_t index_img_sW,
    // depth_img strides
    int64_t depth_img_sN,
    int64_t depth_img_sH,
    int64_t depth_img_sW,
    // bary_img strides
    int64_t bary_img_sN,
    int64_t bary_img_sB,
    int64_t bary_img_sH,
    int64_t bary_img_sW) {
  const int64_t count = N * H * W;

  at::parallel_for(0, count, /*grain_size=*/4096, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t w = index % W;
      const int64_t h = (index / W) % H;
      const int64_t n = index / (H * W);

      const int32_t tr_index =
          index_img_ptr[n * index_img_sN + h * index_img_sH + w * index_img_sW];
      scalar_t* bary_out = bary_img_ptr + bary_img_sN * n + bary_img_sH * h + bary_img_sW * w;
      scalar_t* depth_out = depth_img_ptr + depth_img_sN * n + depth_img_sH * h + depth_img_sW * w;

      if (tr_index != -1) {
        const int32_t* vi_face = vi_ptr + n * vi_sN + tr_index * vi_sV;
        const int32_t vi_0 = vi_face[0 * vi_sF];
        const int32_t vi_1 = vi_face[1 * vi_sF];
        const int32_t vi_2 = vi_face[2 * vi_sF];

        const scalar_t* v_n = v_ptr + n * v_sN;
        const scalar_t p0_x = v_n[v_sV * vi_0 + v_sC * 0];
        const scalar_t p0_y = v_n[v_sV * vi_0 + v_sC * 1];
        const scalar_t p1_x = v_n[v_sV * vi_1 + v_sC * 0];
        const scalar_t p1_y = v_n[v_sV * vi_1 + v_sC * 1];
        const scalar_t p2_x = v_n[v_sV * vi_2 + v_sC * 0];
        const scalar_t p2_y = v_n[v_sV * vi_2 + v_sC * 1];

        const scalar_t p0_z = v_n[v_sV * vi_0 + v_sC * 2];
        const scalar_t p1_z = v_n[v_sV * vi_1 + v_sC * 2];
        const scalar_t p2_z = v_n[v_sV * vi_2 + v_sC * 2];

        const scalar_t v01_x = p1_x - p0_x;
        const scalar_t v01_y = p1_y - p0_y;
        const scalar_t v02_x = p2_x - p0_x;
        const scalar_t v02_y = p2_y - p0_y;

        const scalar_t denominator = epsclamp(v01_x * v02_y - v01_y * v02_x);

        const scalar_t vp0_x = static_cast<scalar_t>(w) - p0_x;
        const scalar_t vp0_y = static_cast<scalar_t>(h) - p0_y;

        const scalar_t bary_1_pre = vp0_x * v02_y - vp0_y * v02_x;
        const scalar_t bary_2_pre = vp0_y * v01_x - vp0_x * v01_y;

        const scalar_t bary_1 = bary_1_pre / denominator;
        const scalar_t bary_2 = bary_2_pre / denominator;
        const scalar_t bary_0 = scalar_t(1.0) - bary_1 - bary_2;

        const scalar_t p0_z_eps = epsclamp(p0_z);
        const scalar_t p1_z_eps = epsclamp(p1_z);
        const scalar_t p2_z_eps = epsclamp(p2_z);

        const scalar_t d_inv_0 = scalar_t(1.0) / p0_z_eps;
        const scalar_t d_inv_1 = scalar_t(1.0) / p1_z_eps;
        const scalar_t d_inv_2 = scalar_t(1.0) / p2_z_eps;

        const scalar_t depth_inverse = d_inv_0 * bary_0 + d_inv_1 * bary_1 + d_inv_2 * bary_2;
        const scalar_t depth = scalar_t(1.0) / epsclamp(depth_inverse);

        bary_out[bary_img_sB * 0] = d_inv_0 * bary_0 * depth;
        bary_out[bary_img_sB * 1] = d_inv_1 * bary_1 * depth;
        bary_out[bary_img_sB * 2] = d_inv_2 * bary_2 * depth;
        *depth_out = depth;
      } else {
        bary_out[bary_img_sB * 0] = scalar_t(0);
        bary_out[bary_img_sB * 1] = scalar_t(0);
        bary_out[bary_img_sB * 2] = scalar_t(0);
        *depth_out = scalar_t(0);
      }
    }
  });
}

template <typename scalar_t>
void render_backward_cpu_impl(
    const scalar_t* v_ptr,
    const int32_t* vi_ptr,
    const int32_t* index_img_ptr,
    const scalar_t* grad_depth_img_ptr,
    const scalar_t* grad_bary_img_ptr,
    scalar_t* grad_v_ptr,
    int64_t N,
    int64_t V,
    int64_t H,
    int64_t W,
    // v strides
    int64_t v_sN,
    int64_t v_sV,
    int64_t v_sC,
    // vi strides
    int64_t vi_sN,
    int64_t vi_sV,
    int64_t vi_sF,
    // index_img strides
    int64_t index_img_sN,
    int64_t index_img_sH,
    int64_t index_img_sW,
    // grad_depth_img strides
    int64_t grad_depth_img_sN,
    int64_t grad_depth_img_sH,
    int64_t grad_depth_img_sW,
    // grad_bary_img strides
    int64_t grad_bary_img_sN,
    int64_t grad_bary_img_sB,
    int64_t grad_bary_img_sH,
    int64_t grad_bary_img_sW,
    // grad_v strides
    int64_t grad_v_sN,
    int64_t grad_v_sV,
    int64_t grad_v_sC) {
  // Parallelize over all pixels (N*H*W), matching the CUDA backward kernel.
  // Multiple pixels may reference the same vertex, so gradient accumulation
  // uses atomic_add (CAS loop on std::atomic, lock-free on x86-64/aarch64).
  const int64_t count = N * H * W;

  at::parallel_for(0, count, /*grain_size=*/4096, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t w = index % W;
      const int64_t h = (index / W) % H;
      const int64_t n = index / (H * W);

      const int32_t tr_index =
          index_img_ptr[n * index_img_sN + h * index_img_sH + w * index_img_sW];

      if (tr_index == -1)
        continue;

      const int32_t* vi_face = vi_ptr + n * vi_sN + tr_index * vi_sV;
      const int32_t vi_0 = vi_face[0 * vi_sF];
      const int32_t vi_1 = vi_face[1 * vi_sF];
      const int32_t vi_2 = vi_face[2 * vi_sF];

      const scalar_t* v_n = v_ptr + n * v_sN;
      const scalar_t p0_x = v_n[v_sV * vi_0 + v_sC * 0];
      const scalar_t p0_y = v_n[v_sV * vi_0 + v_sC * 1];
      const scalar_t p1_x = v_n[v_sV * vi_1 + v_sC * 0];
      const scalar_t p1_y = v_n[v_sV * vi_1 + v_sC * 1];
      const scalar_t p2_x = v_n[v_sV * vi_2 + v_sC * 0];
      const scalar_t p2_y = v_n[v_sV * vi_2 + v_sC * 1];

      const scalar_t p0_z = v_n[v_sV * vi_0 + v_sC * 2];
      const scalar_t p1_z = v_n[v_sV * vi_1 + v_sC * 2];
      const scalar_t p2_z = v_n[v_sV * vi_2 + v_sC * 2];

      const scalar_t v01_x = p1_x - p0_x;
      const scalar_t v01_y = p1_y - p0_y;
      const scalar_t v02_x = p2_x - p0_x;
      const scalar_t v02_y = p2_y - p0_y;

      const scalar_t _denominator = v01_x * v02_y - v01_y * v02_x;
      const scalar_t denominator = epsclamp(_denominator);
      const bool denominator_clamped = denominator != _denominator;

      const scalar_t vp0_x = static_cast<scalar_t>(w) - p0_x;
      const scalar_t vp0_y = static_cast<scalar_t>(h) - p0_y;

      const scalar_t bary_1_pre = vp0_x * v02_y - vp0_y * v02_x;
      const scalar_t bary_2_pre = vp0_y * v01_x - vp0_x * v01_y;

      const scalar_t bary_1 = bary_1_pre / denominator;
      const scalar_t bary_2 = bary_2_pre / denominator;
      const scalar_t bary_0 = scalar_t(1.0) - bary_1 - bary_2;

      const scalar_t p0_z_eps = epsclamp(p0_z);
      const scalar_t p1_z_eps = epsclamp(p1_z);
      const scalar_t p2_z_eps = epsclamp(p2_z);

      const bool z0_clamped = p0_z_eps != p0_z;
      const bool z1_clamped = p1_z_eps != p1_z;
      const bool z2_clamped = p2_z_eps != p2_z;

      const scalar_t d_inv_0 = scalar_t(1.0) / p0_z_eps;
      const scalar_t d_inv_1 = scalar_t(1.0) / p1_z_eps;
      const scalar_t d_inv_2 = scalar_t(1.0) / p2_z_eps;

      const scalar_t depth_inverse = d_inv_0 * bary_0 + d_inv_1 * bary_1 + d_inv_2 * bary_2;
      const scalar_t depth_inverse_eps = epsclamp(depth_inverse);
      const bool depth_inverse_clamped = depth_inverse_eps != depth_inverse;
      const scalar_t depth = scalar_t(1.0) / depth_inverse_eps;

      const scalar_t* grad_bary_ptr =
          grad_bary_img_ptr + grad_bary_img_sN * n + grad_bary_img_sH * h + grad_bary_img_sW * w;
      const scalar_t dL_bary3D_0 = grad_bary_ptr[grad_bary_img_sB * 0];
      const scalar_t dL_bary3D_1 = grad_bary_ptr[grad_bary_img_sB * 1];
      const scalar_t dL_bary3D_2 = grad_bary_ptr[grad_bary_img_sB * 2];

      const scalar_t* grad_depth_ptr = grad_depth_img_ptr + grad_depth_img_sN * n +
          grad_depth_img_sH * h + grad_depth_img_sW * w;
      const scalar_t dL_depth = *grad_depth_ptr + dL_bary3D_0 * d_inv_0 * bary_0 +
          dL_bary3D_1 * d_inv_1 * bary_1 + dL_bary3D_2 * d_inv_2 * bary_2;

      const scalar_t dL_depth_inverse =
          depth_inverse_clamped ? scalar_t(0) : (-dL_depth / (depth_inverse * depth_inverse));

      const scalar_t dL_d_inv_0 = dL_bary3D_0 * bary_0 * depth + dL_depth_inverse * bary_0;
      const scalar_t dL_d_inv_1 = dL_bary3D_1 * bary_1 * depth + dL_depth_inverse * bary_1;
      const scalar_t dL_d_inv_2 = dL_bary3D_2 * bary_2 * depth + dL_depth_inverse * bary_2;

      const scalar_t dL_p0_z = -dL_d_inv_0 / (p0_z_eps * p0_z_eps);
      const scalar_t dL_p1_z = -dL_d_inv_1 / (p1_z_eps * p1_z_eps);
      const scalar_t dL_p2_z = -dL_d_inv_2 / (p2_z_eps * p2_z_eps);

      // Accumulate z gradients with atomic add (multiple pixels may share vertices)
      scalar_t* grad_v_n = grad_v_ptr + grad_v_sN * n;
      if (!z0_clamped)
        atomic_add(&grad_v_n[grad_v_sV * vi_0 + grad_v_sC * 2], dL_p0_z);
      if (!z1_clamped)
        atomic_add(&grad_v_n[grad_v_sV * vi_1 + grad_v_sC * 2], dL_p1_z);
      if (!z2_clamped)
        atomic_add(&grad_v_n[grad_v_sV * vi_2 + grad_v_sC * 2], dL_p2_z);

      // Barycentric gradients
      const scalar_t dL_bary_0 = dL_bary3D_0 * d_inv_0 * depth + dL_depth_inverse * d_inv_0;
      const scalar_t dL_bary_1 = dL_bary3D_1 * d_inv_1 * depth + dL_depth_inverse * d_inv_1;
      const scalar_t dL_bary_2 = dL_bary3D_2 * d_inv_2 * depth + dL_depth_inverse * d_inv_2;

      const scalar_t dL_bary12_x = -dL_bary_0 + dL_bary_1;
      const scalar_t dL_bary12_y = -dL_bary_0 + dL_bary_2;
      const scalar_t dL_bary_pre_x = dL_bary12_x / denominator;
      const scalar_t dL_bary_pre_y = dL_bary12_y / denominator;

      const scalar_t dL_denominator =
          denominator_clamped ? scalar_t(0) : -(dL_bary_pre_x * bary_1 + dL_bary_pre_y * bary_2);

      const scalar_t dL_vp0_x = dL_bary_pre_x * v02_y - dL_bary_pre_y * v01_y;
      const scalar_t dL_vp0_y = -dL_bary_pre_x * v02_x + dL_bary_pre_y * v01_x;

      const scalar_t dL_v02_x = -dL_bary_pre_x * vp0_y - dL_denominator * v01_y;
      const scalar_t dL_v02_y = dL_bary_pre_x * vp0_x + dL_denominator * v01_x;
      const scalar_t dL_v01_x = dL_bary_pre_y * vp0_y + dL_denominator * v02_y;
      const scalar_t dL_v01_y = -dL_bary_pre_y * vp0_x - dL_denominator * v02_x;

      const scalar_t dL_p0_x = -dL_v02_x - dL_v01_x - dL_vp0_x;
      const scalar_t dL_p0_y = -dL_v02_y - dL_v01_y - dL_vp0_y;
      const scalar_t dL_p1_x = dL_v01_x;
      const scalar_t dL_p1_y = dL_v01_y;
      const scalar_t dL_p2_x = dL_v02_x;
      const scalar_t dL_p2_y = dL_v02_y;

      atomic_add(&grad_v_n[grad_v_sV * vi_0 + grad_v_sC * 0], dL_p0_x);
      atomic_add(&grad_v_n[grad_v_sV * vi_0 + grad_v_sC * 1], dL_p0_y);
      atomic_add(&grad_v_n[grad_v_sV * vi_1 + grad_v_sC * 0], dL_p1_x);
      atomic_add(&grad_v_n[grad_v_sV * vi_1 + grad_v_sC * 1], dL_p1_y);
      atomic_add(&grad_v_n[grad_v_sV * vi_2 + grad_v_sC * 0], dL_p2_x);
      atomic_add(&grad_v_n[grad_v_sV * vi_2 + grad_v_sC * 1], dL_p2_y);
    }
  });
}

} // namespace

std::vector<torch::Tensor>
render_cpu(const torch::Tensor& v, const torch::Tensor& vi, const torch::Tensor& index_img) {
  TORCH_CHECK(
      v.defined() && vi.defined() && index_img.defined(),
      "render(): expected all inputs to be defined");
  TORCH_CHECK(
      v.device().is_cpu() && vi.device().is_cpu() && index_img.device().is_cpu(),
      "render(): expected all inputs to be on CPU");
  TORCH_CHECK(
      v.is_floating_point(),
      "render(): expected v to have floating point type, but v has ",
      v.dtype());
  TORCH_CHECK(
      vi.dtype() == torch::kInt32,
      "render(): expected vi to have int32 type, but vi has ",
      vi.dtype());
  TORCH_CHECK(
      index_img.dtype() == torch::kInt32,
      "render(): expected index_img to have int32 type, but index_img has ",
      index_img.dtype());
  TORCH_CHECK(
      v.layout() == torch::kStrided && vi.layout() == torch::kStrided &&
          index_img.layout() == torch::kStrided,
      "render(): expected all inputs to have torch.strided layout");
  TORCH_CHECK(
      (v.dim() == 3) && (vi.dim() == 3) && (index_img.dim() == 3),
      "render(): expected v.ndim == 3, vi.ndim == 3, index_img.ndim == 3, "
      "but got v with sizes ",
      v.sizes(),
      " and vi with sizes ",
      vi.sizes(),
      " and index_img with sizes ",
      index_img.sizes());
  TORCH_CHECK(
      v.size(0) == index_img.size(0),
      "render(): expected v and index_img to have same batch size, "
      "but got v with sizes ",
      v.sizes(),
      " and index_img with sizes ",
      index_img.sizes());
  TORCH_CHECK(
      vi.size(0) == v.size(0),
      "render(): expected first dim of vi to match first dim of v but got ",
      v.size(0),
      " in first dim of v, and ",
      vi.size(0),
      " in the first dim of vi");
  TORCH_CHECK(
      v.size(2) == 3 && vi.size(2) == 3,
      "render(): expected third dim of v and vi to be 3, but got ",
      v.size(2),
      " and ",
      vi.size(2));

  auto v_c = v.contiguous();
  auto vi_c = vi.contiguous();
  auto index_img_c = index_img.contiguous();

  const auto N = v_c.size(0);
  const auto H = index_img_c.size(1);
  const auto W = index_img_c.size(2);
  const auto V = v_c.size(1);
  const int64_t count = N * H * W;

  auto depth_img = at::empty({N, H, W}, v.options());
  auto bary_img = at::empty({N, 3, H, W}, v.options());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(v_c.scalar_type(), "render_cpu", [&] {
      render_forward_cpu_impl<scalar_t>(
          v_c.data_ptr<scalar_t>(),
          vi_c.data_ptr<int32_t>(),
          index_img_c.data_ptr<int32_t>(),
          depth_img.data_ptr<scalar_t>(),
          bary_img.data_ptr<scalar_t>(),
          N,
          V,
          H,
          W,
          /*v_sN=*/V * 3,
          /*v_sV=*/3,
          /*v_sC=*/1,
          /*vi_sN=*/vi_c.size(1) * 3,
          /*vi_sV=*/3,
          /*vi_sF=*/1,
          /*index_img_sN=*/H * W,
          /*index_img_sH=*/W,
          /*index_img_sW=*/1,
          /*depth_img_sN=*/H * W,
          /*depth_img_sH=*/W,
          /*depth_img_sW=*/1,
          /*bary_img_sN=*/3 * H * W,
          /*bary_img_sB=*/H * W,
          /*bary_img_sH=*/W,
          /*bary_img_sW=*/1);
    });
  }
  return {depth_img, bary_img};
}

torch::Tensor render_cpu_backward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& grad_depth_img,
    const torch::Tensor& grad_bary_img) {
  auto v_c = v.contiguous();
  auto vi_c = vi.contiguous();
  auto index_img_c = index_img.contiguous();
  auto grad_depth_c = grad_depth_img.contiguous();
  auto grad_bary_c = grad_bary_img.contiguous();

  const auto N = v_c.size(0);
  const auto V = v_c.size(1);
  const auto C = v_c.size(2);
  const auto H = index_img_c.size(1);
  const auto W = index_img_c.size(2);
  const int64_t count = N * H * W;

  auto grad_v = at::zeros({N, V, C}, v.options());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(v_c.scalar_type(), "render_cpu_backward", [&] {
      render_backward_cpu_impl<scalar_t>(
          v_c.data_ptr<scalar_t>(),
          vi_c.data_ptr<int32_t>(),
          index_img_c.data_ptr<int32_t>(),
          grad_depth_c.data_ptr<scalar_t>(),
          grad_bary_c.data_ptr<scalar_t>(),
          grad_v.data_ptr<scalar_t>(),
          N,
          V,
          H,
          W,
          /*v_sN=*/V * 3,
          /*v_sV=*/3,
          /*v_sC=*/1,
          /*vi_sN=*/vi_c.size(1) * 3,
          /*vi_sV=*/3,
          /*vi_sF=*/1,
          /*index_img_sN=*/H * W,
          /*index_img_sH=*/W,
          /*index_img_sW=*/1,
          /*grad_depth_img_sN=*/H * W,
          /*grad_depth_img_sH=*/W,
          /*grad_depth_img_sW=*/1,
          /*grad_bary_img_sN=*/3 * H * W,
          /*grad_bary_img_sB=*/H * W,
          /*grad_bary_img_sH=*/W,
          /*grad_bary_img_sW=*/1,
          /*grad_v_sN=*/V * C,
          /*grad_v_sV=*/C,
          /*grad_v_sC=*/1);
    });
  }
  return grad_v;
}

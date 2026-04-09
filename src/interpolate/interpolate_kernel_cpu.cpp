// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Parallel.h>
#include <cpu_atomic.h>
#include <torch/types.h>

#include "interpolate_kernel.h"

using drtk::atomic_add;

namespace {

template <typename scalar_t>
void interpolate_forward_cpu_impl(
    const scalar_t* vert_attr_ptr,
    const int32_t* vi_ptr,
    const int32_t* index_img_ptr,
    const scalar_t* bary_img_ptr,
    scalar_t* out_img_ptr,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    // vert_attributes strides
    int64_t va_sN,
    int64_t va_sV,
    int64_t va_sC,
    // vi strides
    int64_t vi_sN,
    int64_t vi_sV,
    int64_t vi_sF,
    // index_img strides
    int64_t idx_sN,
    int64_t idx_sH,
    int64_t idx_sW,
    // bary_img strides
    int64_t bary_sN,
    int64_t bary_sB,
    int64_t bary_sH,
    int64_t bary_sW,
    // out_img strides
    int64_t out_sN,
    int64_t out_sC,
    int64_t out_sH,
    int64_t out_sW) {
  const int64_t count = N * H * W;

  at::parallel_for(0, count, /*grain_size=*/4096, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t w = index % W;
      const int64_t h = (index / W) % H;
      const int64_t n = index / (H * W);

      const int32_t tr_index = index_img_ptr[n * idx_sN + h * idx_sH + w * idx_sW];
      scalar_t* out_ptr = out_img_ptr + out_sN * n + out_sH * h + out_sW * w;

      if (tr_index != -1) {
        const int32_t* vi_face = vi_ptr + n * vi_sN + tr_index * vi_sV;
        const int32_t vi_0 = vi_face[0 * vi_sF];
        const int32_t vi_1 = vi_face[1 * vi_sF];
        const int32_t vi_2 = vi_face[2 * vi_sF];

        const scalar_t* vert_n = vert_attr_ptr + va_sN * n;
        const scalar_t* vert_0 = vert_n + va_sV * vi_0;
        const scalar_t* vert_1 = vert_n + va_sV * vi_1;
        const scalar_t* vert_2 = vert_n + va_sV * vi_2;

        const scalar_t* bary_ptr = bary_img_ptr + bary_sN * n + bary_sH * h + bary_sW * w;
        const scalar_t bary_0 = bary_ptr[0 * bary_sB];
        const scalar_t bary_1 = bary_ptr[1 * bary_sB];
        const scalar_t bary_2 = bary_ptr[2 * bary_sB];

        for (int64_t i = 0; i < C; ++i) {
          scalar_t v0 = vert_0[i * va_sC];
          scalar_t v1 = vert_1[i * va_sC];
          scalar_t v2 = vert_2[i * va_sC];
          out_ptr[out_sC * i] = v0 * bary_0 + v1 * bary_1 + v2 * bary_2;
        }
      } else {
        const scalar_t bg[2] = {
            (w * scalar_t(2.0) + scalar_t(1.0)) / W - scalar_t(1.0),
            (h * scalar_t(2.0) + scalar_t(1.0)) / H - scalar_t(1.0)};
        for (int64_t i = 0; i < C; ++i) {
          out_ptr[out_sC * i] = bg[i % 2];
        }
      }
    }
  });
}

template <typename scalar_t>
void interpolate_backward_cpu_impl(
    const scalar_t* grad_out_ptr,
    const scalar_t* vert_attr_ptr,
    const int32_t* vi_ptr,
    const int32_t* index_img_ptr,
    const scalar_t* bary_img_ptr,
    scalar_t* vert_grad_ptr,
    scalar_t* bary_grad_ptr,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    // vert_attributes strides
    int64_t va_sN,
    int64_t va_sV,
    int64_t va_sC,
    // vi strides
    int64_t vi_sN,
    int64_t vi_sV,
    int64_t vi_sF,
    // index_img strides
    int64_t idx_sN,
    int64_t idx_sH,
    int64_t idx_sW,
    // bary_img strides
    int64_t bary_sN,
    int64_t bary_sB,
    int64_t bary_sH,
    int64_t bary_sW,
    // grad_out strides
    int64_t go_sN,
    int64_t go_sC,
    int64_t go_sH,
    int64_t go_sW,
    // vert_grad strides
    int64_t vg_sN,
    int64_t vg_sV,
    int64_t vg_sC,
    // bary_grad strides
    int64_t bg_sN,
    int64_t bg_sB,
    int64_t bg_sH,
    int64_t bg_sW,
    bool bary_requires_grad,
    bool vert_requires_grad) {
  const int64_t count = N * H * W;

  // Parallelize over all pixels (N*H*W), matching the CUDA backward kernel.
  // Gradient accumulation into vert_grad uses atomic_add since multiple
  // pixels may reference the same vertex.
  at::parallel_for(0, count, /*grain_size=*/4096, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t w = index % W;
      const int64_t h = (index / W) % H;
      const int64_t n = index / (H * W);

      const int32_t tr_index = index_img_ptr[n * idx_sN + h * idx_sH + w * idx_sW];
      const scalar_t* go_ptr = grad_out_ptr + go_sN * n + go_sH * h + go_sW * w;

      if (tr_index != -1) {
        const int32_t* vi_face = vi_ptr + n * vi_sN + tr_index * vi_sV;
        const int32_t vi_0 = vi_face[0 * vi_sF];
        const int32_t vi_1 = vi_face[1 * vi_sF];
        const int32_t vi_2 = vi_face[2 * vi_sF];

        const scalar_t* vert_n = vert_attr_ptr + va_sN * n;
        const scalar_t* vert_0 = vert_n + va_sV * vi_0;
        const scalar_t* vert_1 = vert_n + va_sV * vi_1;
        const scalar_t* vert_2 = vert_n + va_sV * vi_2;

        scalar_t* vert_grad_n = vert_requires_grad ? vert_grad_ptr + vg_sN * n : nullptr;

        const scalar_t* bary_ptr = bary_img_ptr + bary_sN * n + bary_sH * h + bary_sW * w;
        scalar_t bary_0 = scalar_t(0), bary_1 = scalar_t(0), bary_2 = scalar_t(0);
        if (vert_requires_grad) {
          bary_0 = bary_ptr[0 * bary_sB];
          bary_1 = bary_ptr[1 * bary_sB];
          bary_2 = bary_ptr[2 * bary_sB];
        }

        scalar_t bg_0 = scalar_t(0), bg_1 = scalar_t(0), bg_2 = scalar_t(0);

        for (int64_t i = 0; i < C; ++i) {
          scalar_t g_out = go_ptr[i * go_sC];

          if (bary_requires_grad) {
            scalar_t v0 = vert_0[i * va_sC];
            scalar_t v1 = vert_1[i * va_sC];
            scalar_t v2 = vert_2[i * va_sC];
            bg_0 += g_out * v0;
            bg_1 += g_out * v1;
            bg_2 += g_out * v2;
          }

          if (vert_requires_grad) {
            atomic_add(&vert_grad_n[vg_sV * vi_0 + vg_sC * i], g_out * bary_0);
            atomic_add(&vert_grad_n[vg_sV * vi_1 + vg_sC * i], g_out * bary_1);
            atomic_add(&vert_grad_n[vg_sV * vi_2 + vg_sC * i], g_out * bary_2);
          }
        }

        if (bary_requires_grad) {
          scalar_t* bg_ptr = bary_grad_ptr + bg_sN * n + bg_sH * h + bg_sW * w;
          bg_ptr[0 * bg_sB] = bg_0;
          bg_ptr[1 * bg_sB] = bg_1;
          bg_ptr[2 * bg_sB] = bg_2;
        }
      } else {
        if (bary_requires_grad) {
          scalar_t* bg_ptr = bary_grad_ptr + bg_sN * n + bg_sH * h + bg_sW * w;
          bg_ptr[0 * bg_sB] = scalar_t(0);
          bg_ptr[1 * bg_sB] = scalar_t(0);
          bg_ptr[2 * bg_sB] = scalar_t(0);
        }
      }
    }
  });
}

} // namespace

torch::Tensor interpolate_cpu(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  TORCH_CHECK(
      vert_attributes.defined() && vi.defined() && index_img.defined() && bary_img.defined(),
      "interpolate(): expected all inputs to be defined");
  TORCH_CHECK(
      vert_attributes.device().is_cpu() && vi.device().is_cpu() && index_img.device().is_cpu() &&
          bary_img.device().is_cpu(),
      "interpolate(): expected all inputs to be on CPU");
  TORCH_CHECK(
      vert_attributes.dtype() == bary_img.dtype(),
      "interpolate(): expected vert_attributes and bary_img to have same dtype, but vert_attributes has ",
      vert_attributes.dtype(),
      " and bary_img has ",
      bary_img.dtype());
  TORCH_CHECK(
      vert_attributes.is_floating_point(),
      "interpolate(): expected vert_attributes to have floating point type, but has ",
      vert_attributes.dtype());
  TORCH_CHECK(
      vi.dtype() == torch::kInt32,
      "interpolate(): expected vi to have int32 type, but vi has ",
      vi.dtype());
  TORCH_CHECK(
      index_img.dtype() == torch::kInt32,
      "interpolate(): expected index_img to have int32 type, but index_img has ",
      index_img.dtype());
  TORCH_CHECK(
      vert_attributes.layout() == torch::kStrided && vi.layout() == torch::kStrided &&
          index_img.layout() == torch::kStrided && bary_img.layout() == torch::kStrided,
      "interpolate(): expected all inputs to have torch.strided layout");
  TORCH_CHECK(
      (vert_attributes.dim() == 3) && (vi.dim() == 3) && (index_img.dim() == 3) &&
          (bary_img.dim() == 4),
      "interpolate(): expected vert_attributes.ndim == 3, vi.ndim == 3, index_img.ndim == 3, bary_img.ndim == 4, "
      "but got vert_attributes with sizes ",
      vert_attributes.sizes(),
      " and vi with sizes ",
      vi.sizes(),
      " and index_img with sizes ",
      index_img.sizes(),
      " and bary_img with sizes ",
      bary_img.sizes());
  TORCH_CHECK(
      vert_attributes.size(0) == index_img.size(0) && vert_attributes.size(0) == bary_img.size(0),
      "interpolate(): expected vert_attributes, index_img and bary_img to have same batch size");
  TORCH_CHECK(
      vi.size(2) == 3 && bary_img.size(1) == 3,
      "interpolate(): expected last dim of vi to be 3 and second dim of bary_img to be 3");
  TORCH_CHECK(
      vi.size(0) == vert_attributes.size(0),
      "interpolate(): expected vi to have same first dimension as vert_attributes");
  TORCH_CHECK(
      index_img.size(1) == bary_img.size(2) && index_img.size(2) == bary_img.size(3),
      "interpolate(): expected H and W dims of index_img and bary_img to match");

  auto va_c = vert_attributes.contiguous();
  auto vi_c = vi.contiguous();
  auto index_img_c = index_img.contiguous();
  auto bary_img_c = bary_img.contiguous();

  const auto N = va_c.size(0);
  const auto V = va_c.size(1);
  const auto C = va_c.size(2);
  const auto H = bary_img_c.size(2);
  const auto W = bary_img_c.size(3);
  const int64_t count = N * H * W;

  auto output = at::empty({N, C, H, W}, vert_attributes.options());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(va_c.scalar_type(), "interpolate_cpu", [&] {
      interpolate_forward_cpu_impl<scalar_t>(
          va_c.data_ptr<scalar_t>(),
          vi_c.data_ptr<int32_t>(),
          index_img_c.data_ptr<int32_t>(),
          bary_img_c.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          N,
          C,
          H,
          W,
          /*va_sN=*/V * C,
          /*va_sV=*/C,
          /*va_sC=*/1,
          /*vi_sN=*/vi_c.size(1) * 3,
          /*vi_sV=*/3,
          /*vi_sF=*/1,
          /*idx_sN=*/H * W,
          /*idx_sH=*/W,
          /*idx_sW=*/1,
          /*bary_sN=*/3 * H * W,
          /*bary_sB=*/H * W,
          /*bary_sH=*/W,
          /*bary_sW=*/1,
          /*out_sN=*/C * H * W,
          /*out_sC=*/H * W,
          /*out_sH=*/W,
          /*out_sW=*/1);
    });
  }
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> interpolate_cpu_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  auto va_c = vert_attributes.contiguous();
  auto vi_c = vi.contiguous();
  auto index_img_c = index_img.contiguous();
  auto bary_img_c = bary_img.contiguous();
  auto grad_out_c = grad_out.contiguous();

  const auto N = va_c.size(0);
  const auto V = va_c.size(1);
  const auto C = va_c.size(2);
  const auto H = bary_img_c.size(2);
  const auto W = bary_img_c.size(3);
  const int64_t count = N * H * W;

  bool bary_requires_grad = bary_img.requires_grad();
  bool vert_requires_grad = vert_attributes.requires_grad();

  auto vert_grad =
      vert_requires_grad ? at::zeros({N, V, C}, vert_attributes.options()) : torch::Tensor();
  auto bary_grad =
      bary_requires_grad ? at::empty({N, 3, H, W}, bary_img.options()) : torch::Tensor();

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(va_c.scalar_type(), "interpolate_cpu_backward", [&] {
      interpolate_backward_cpu_impl<scalar_t>(
          grad_out_c.data_ptr<scalar_t>(),
          va_c.data_ptr<scalar_t>(),
          vi_c.data_ptr<int32_t>(),
          index_img_c.data_ptr<int32_t>(),
          bary_img_c.data_ptr<scalar_t>(),
          vert_requires_grad ? vert_grad.data_ptr<scalar_t>() : nullptr,
          bary_requires_grad ? bary_grad.data_ptr<scalar_t>() : nullptr,
          N,
          C,
          H,
          W,
          /*va_sN=*/V * C,
          /*va_sV=*/C,
          /*va_sC=*/1,
          /*vi_sN=*/vi_c.size(1) * 3,
          /*vi_sV=*/3,
          /*vi_sF=*/1,
          /*idx_sN=*/H * W,
          /*idx_sH=*/W,
          /*idx_sW=*/1,
          /*bary_sN=*/3 * H * W,
          /*bary_sB=*/H * W,
          /*bary_sH=*/W,
          /*bary_sW=*/1,
          /*go_sN=*/C * H * W,
          /*go_sC=*/H * W,
          /*go_sH=*/W,
          /*go_sW=*/1,
          /*vg_sN=*/V * C,
          /*vg_sV=*/C,
          /*vg_sC=*/1,
          /*bg_sN=*/3 * H * W,
          /*bg_sB=*/H * W,
          /*bg_sH=*/W,
          /*bg_sW=*/1,
          bary_requires_grad,
          vert_requires_grad);
    });
  }
  return std::make_tuple(vert_grad, bary_grad);
}

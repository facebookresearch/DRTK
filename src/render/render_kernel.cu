// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <torch/types.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include "render_kernel.h"

#include <kernel_utils.h>

using namespace math;

using at::native::fastAtomicAdd;

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void render_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> v,
    TensorInfo<int32_t, index_t> vi,
    TensorInfo<int32_t, index_t> index_img,
    TensorInfo<scalar_t, index_t> depth_img,
    TensorInfo<scalar_t, index_t> bary_img) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;

  const index_t H = bary_img.sizes[2];
  const index_t W = bary_img.sizes[3];
  const index_t V = v.sizes[1];

  const index_t v_sN = v.strides[0];
  const index_t v_sV = v.strides[1];
  const index_t v_sC = v.strides[2];

  const index_t vi_sV = vi.strides[0];
  const index_t vi_sF = vi.strides[1];

  const index_t index_img_sN = index_img.strides[0];
  const index_t index_img_sH = index_img.strides[1];
  const index_t index_img_sW = index_img.strides[2];

  const index_t depth_img_sN = depth_img.strides[0];
  const index_t depth_img_sH = depth_img.strides[1];
  const index_t depth_img_sW = depth_img.strides[2];

  const index_t bary_img_sN = bary_img.strides[0];
  const index_t bary_img_sB = bary_img.strides[1];
  const index_t bary_img_sH = bary_img.strides[2];
  const index_t bary_img_sW = bary_img.strides[3];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t n = index / (H * W);

    const int32_t tr_index = index_img.data[n * index_img_sN + h * index_img_sH + w * index_img_sW];
    scalar_t* __restrict bary_img_ptr =
        bary_img.data + bary_img_sN * n + bary_img_sH * h + bary_img_sW * w;
    scalar_t* __restrict depth_img_ptr =
        depth_img.data + depth_img_sN * n + depth_img_sH * h + depth_img_sW * w;

    if (tr_index != -1) {
      const int32_t* __restrict vi_ptr = vi.data + tr_index * vi_sV;
      const int32_t vi_0 = vi_ptr[0 * vi_sF];
      const int32_t vi_1 = vi_ptr[1 * vi_sF];
      const int32_t vi_2 = vi_ptr[2 * vi_sF];

      assert(vi_0 < V && vi_1 < V && vi_2 < V);

      const scalar_t* __restrict v_ptr = v.data + n * v_sN;
      const scalar2_t p_0 = {v_ptr[v_sV * vi_0 + v_sC * 0], v_ptr[v_sV * vi_0 + v_sC * 1]};
      const scalar2_t p_1 = {v_ptr[v_sV * vi_1 + v_sC * 0], v_ptr[v_sV * vi_1 + v_sC * 1]};
      const scalar2_t p_2 = {v_ptr[v_sV * vi_2 + v_sC * 0], v_ptr[v_sV * vi_2 + v_sC * 1]};

      const scalar3_t p_012_z = {
          v_ptr[v_sV * vi_0 + v_sC * 2],
          v_ptr[v_sV * vi_1 + v_sC * 2],
          v_ptr[v_sV * vi_2 + v_sC * 2]};

      const scalar2_t v_01 = p_1 - p_0;
      const scalar2_t v_02 = p_2 - p_0;
      const scalar_t denominator = epsclamp((v_01.x * v_02.y - v_01.y * v_02.x));

      const scalar2_t vp0p = {w - p_0.x, h - p_0.y};

      const scalar2_t bary_12_pre = scalar2_t{
          (vp0p.x * v_02.y - vp0p.y * v_02.x),
          (vp0p.y * v_01.x - vp0p.x * v_01.y),
      };
      const scalar2_t bary_12 = bary_12_pre / denominator;
      scalar3_t bary = {scalar_t(1.0) - bary_12.x - bary_12.y, bary_12.x, bary_12.y};

      const scalar3_t p_012_z_eps = epsclamp(p_012_z);
      const scalar3_t d_inv = 1.0 / p_012_z_eps;

      const scalar_t depth_inverse = dot(d_inv, bary);
      const scalar_t depth = 1.0f / epsclamp(depth_inverse);

      const scalar3_t bary_3D = d_inv * bary * depth;
      bary_img_ptr[bary_img_sB * 0] = bary_3D.x;
      bary_img_ptr[bary_img_sB * 1] = bary_3D.y;
      bary_img_ptr[bary_img_sB * 2] = bary_3D.z;
      *depth_img_ptr = depth;
    } else {
      bary_img_ptr[bary_img_sB * 0] = scalar_t(0);
      bary_img_ptr[bary_img_sB * 1] = scalar_t(0);
      bary_img_ptr[bary_img_sB * 2] = scalar_t(0);
      *depth_img_ptr = scalar_t(0);
    }
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void render_backward_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> v,
    TensorInfo<int32_t, index_t> vi,
    TensorInfo<int32_t, index_t> index_img,
    TensorInfo<scalar_t, index_t> grad_depth_img,
    TensorInfo<scalar_t, index_t> grad_bary_img,
    TensorInfo<scalar_t, index_t> grad_v,
    const index_t memory_span) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;

  const index_t H = grad_bary_img.sizes[2];
  const index_t W = grad_bary_img.sizes[3];
  const index_t V = v.sizes[1];

  const index_t v_sN = v.strides[0];
  const index_t v_sV = v.strides[1];
  const index_t v_sC = v.strides[2];

  const index_t vi_sV = vi.strides[0];
  const index_t vi_sF = vi.strides[1];

  const index_t index_img_sN = index_img.strides[0];
  const index_t index_img_sH = index_img.strides[1];
  const index_t index_img_sW = index_img.strides[2];

  const index_t grad_depth_img_sN = grad_depth_img.strides[0];
  const index_t grad_depth_img_sH = grad_depth_img.strides[1];
  const index_t grad_depth_img_sW = grad_depth_img.strides[2];

  const index_t grad_bary_img_sN = grad_bary_img.strides[0];
  const index_t grad_bary_img_sB = grad_bary_img.strides[1];
  const index_t grad_bary_img_sH = grad_bary_img.strides[2];
  const index_t grad_bary_img_sW = grad_bary_img.strides[3];

  const index_t grad_v_sN = grad_v.strides[0];
  const index_t grad_v_sV = grad_v.strides[1];
  const index_t grad_v_sC = grad_v.strides[2];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t n = index / (H * W);

    const int32_t tr_index = index_img.data[n * index_img_sN + h * index_img_sH + w * index_img_sW];
    const scalar_t* __restrict grad_bary_img_ptr =
        grad_bary_img.data + grad_bary_img_sN * n + grad_bary_img_sH * h + grad_bary_img_sW * w;
    const scalar_t* __restrict grad_depth_img_ptr =
        grad_depth_img.data + grad_depth_img_sN * n + grad_depth_img_sH * h + grad_depth_img_sW * w;

    scalar_t* __restrict grad_v_ptr = grad_v.data + grad_v_sN * n;

    if (tr_index != -1) {
      const int32_t* __restrict vi_ptr = vi.data + tr_index * vi_sV;
      const int32_t vi_0 = vi_ptr[0 * vi_sF];
      const int32_t vi_1 = vi_ptr[1 * vi_sF];
      const int32_t vi_2 = vi_ptr[2 * vi_sF];

      assert(vi_0 < V && vi_1 < V && vi_2 < V);

      const scalar_t* __restrict v_ptr = v.data + n * v_sN;
      const scalar2_t p_0 = {v_ptr[v_sV * vi_0 + v_sC * 0], v_ptr[v_sV * vi_0 + v_sC * 1]};
      const scalar2_t p_1 = {v_ptr[v_sV * vi_1 + v_sC * 0], v_ptr[v_sV * vi_1 + v_sC * 1]};
      const scalar2_t p_2 = {v_ptr[v_sV * vi_2 + v_sC * 0], v_ptr[v_sV * vi_2 + v_sC * 1]};

      const scalar3_t p_012_z = {
          v_ptr[v_sV * vi_0 + v_sC * 2],
          v_ptr[v_sV * vi_1 + v_sC * 2],
          v_ptr[v_sV * vi_2 + v_sC * 2]};

      const scalar2_t v_01 = p_1 - p_0;
      const scalar2_t v_02 = p_2 - p_0;
      const scalar_t _denominator = v_01.x * v_02.y - v_01.y * v_02.x;
      const scalar_t denominator = epsclamp(_denominator);
      const bool denominator_clamped = denominator != _denominator;

      const scalar2_t vp0p = {w - p_0.x, h - p_0.y};

      const scalar2_t bary_12_pre = scalar2_t{
          vp0p.x * v_02.y - vp0p.y * v_02.x,
          vp0p.y * v_01.x - vp0p.x * v_01.y,
      };
      const scalar2_t bary_12 = bary_12_pre / denominator;
      scalar3_t bary = {scalar_t(1.0) - bary_12.x - bary_12.y, bary_12.x, bary_12.y};

      const scalar3_t p_012_z_eps = epsclamp(p_012_z);

      const bool z0_clamped = p_012_z_eps.x != p_012_z.x;
      const bool z1_clamped = p_012_z_eps.y != p_012_z.y;
      const bool z2_clamped = p_012_z_eps.z != p_012_z.z;

      const scalar3_t d_inv = 1.0 / p_012_z_eps;

      const scalar_t depth_inverse = dot(d_inv, bary);
      const scalar_t depth_inverse_eps = epsclamp(depth_inverse);
      const bool depth_inverse_clamped = depth_inverse_eps != depth_inverse;
      const scalar_t depth = 1.0f / depth_inverse_eps;

      const scalar3_t dL_bary_3D = {
          grad_bary_img_ptr[grad_bary_img_sB * 0],
          grad_bary_img_ptr[grad_bary_img_sB * 1],
          grad_bary_img_ptr[grad_bary_img_sB * 2]};
      const scalar_t dL_depth = *grad_depth_img_ptr + dot(dL_bary_3D * d_inv, bary);

      const scalar_t dL_depth_inverse =
          depth_inverse_clamped ? 0.f : (-dL_depth / (depth_inverse * depth_inverse));
      const scalar3_t dL_d_inv = dL_bary_3D * bary * depth + dL_depth_inverse * bary;
      const scalar3_t dL_p_012_z = -dL_d_inv / (p_012_z_eps * p_012_z_eps);

      fastAtomicAdd(
          grad_v_ptr,
          grad_v_sV * vi_0 + grad_v_sC * 2,
          memory_span,
          z0_clamped ? 0.f : dL_p_012_z.x,
          true);
      fastAtomicAdd(
          grad_v_ptr,
          grad_v_sV * vi_1 + grad_v_sC * 2,
          memory_span,
          z1_clamped ? 0.f : dL_p_012_z.y,
          true);
      fastAtomicAdd(
          grad_v_ptr,
          grad_v_sV * vi_2 + grad_v_sC * 2,
          memory_span,
          z2_clamped ? 0.f : dL_p_012_z.z,
          true);

      const scalar3_t dL_bary = dL_bary_3D * d_inv * depth + dL_depth_inverse * d_inv;
      const scalar2_t dL_bary_12 = {-dL_bary.x + dL_bary.y, -dL_bary.x + dL_bary.z};
      const scalar2_t dL_bary_pre = dL_bary_12 / denominator;

      const scalar_t dL_denominator = denominator_clamped ? 0.f : -dot(dL_bary_pre, bary_12);

      const scalar2_t dL_vp0p = {
          dL_bary_pre.x * v_02.y - dL_bary_pre.y * v_01.y,
          -dL_bary_pre.x * v_02.x + dL_bary_pre.y * v_01.x};

      const scalar2_t dL_v_02 = {
          -dL_bary_pre.x * vp0p.y - dL_denominator * v_01.y,
          dL_bary_pre.x * vp0p.x + dL_denominator * v_01.x};
      const scalar2_t dL_v_01 = {
          dL_bary_pre.y * vp0p.y + dL_denominator * v_02.y,
          -dL_bary_pre.y * vp0p.x - dL_denominator * v_02.x};

      const scalar2_t dL_p0 = -dL_v_02 - dL_v_01 - dL_vp0p;
      const scalar2_t dL_p1 = dL_v_01;
      const scalar2_t dL_p2 = dL_v_02;

      fastAtomicAdd(grad_v_ptr, grad_v_sV * vi_0 + grad_v_sC * 0, memory_span, dL_p0.x, true);
      fastAtomicAdd(grad_v_ptr, grad_v_sV * vi_0 + grad_v_sC * 1, memory_span, dL_p0.y, true);
      fastAtomicAdd(grad_v_ptr, grad_v_sV * vi_1 + grad_v_sC * 0, memory_span, dL_p1.x, true);
      fastAtomicAdd(grad_v_ptr, grad_v_sV * vi_1 + grad_v_sC * 1, memory_span, dL_p1.y, true);
      fastAtomicAdd(grad_v_ptr, grad_v_sV * vi_2 + grad_v_sC * 0, memory_span, dL_p2.x, true);
      fastAtomicAdd(grad_v_ptr, grad_v_sV * vi_2 + grad_v_sC * 1, memory_span, dL_p2.y, true);
    }
  }
}

std::vector<torch::Tensor>
render_cuda(const torch::Tensor& v, const torch::Tensor& vi, const torch::Tensor& index_img) {
  TORCH_CHECK(
      v.defined() && vi.defined() && index_img.defined(),
      "render(): expected all inputs to be defined");
  auto v_opt = v.options();
  auto vi_opt = vi.options();
  auto index_img_opt = index_img.options();
  TORCH_CHECK(
      (v.device() == vi.device()) && (v.device() == index_img.device()) && (v.is_cuda()),
      "render(): expected all inputs to be on same cuda device");
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
      (v.dim() == 3) && (vi.dim() == 2) && (index_img.dim() == 3),
      "render(): expected v.ndim == 3, vi.ndim == 2, index_img.ndim == 3, "
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
      v.size(2) == 3 && vi.size(1) == 3,
      "render(): expected third dim of v to be of size 3, and second dim of vi to be of size 3, but got ",
      v.size(2),
      " in the third dim of v, and ",
      vi.size(1),
      " in the second dim of vi");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(v));

  auto N = v.size(0);
  auto H = index_img.size(1);
  auto W = index_img.size(2);
  int64_t count = N * H * W;

  auto depth_img = at::empty({N, H, W}, v.options());
  auto bary_img = at::empty({N, 3, H, W}, v.options());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "render_kernel", [&] {
      if (at::native::canUse32BitIndexMath(v) && at::native::canUse32BitIndexMath(bary_img) &&
          at::native::canUse32BitIndexMath(depth_img) &&
          at::native::canUse32BitIndexMath(index_img) && at::native::canUse32BitIndexMath(vi)) {
        typedef int index_type;

        render_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(count),
                getTensorInfo<scalar_t, index_type>(v),
                getTensorInfo<int32_t, index_type>(vi),
                getTensorInfo<int32_t, index_type>(index_img),
                getTensorInfo<scalar_t, index_type>(depth_img),
                getTensorInfo<scalar_t, index_type>(bary_img));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;

        render_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(count),
                getTensorInfo<scalar_t, index_type>(v),
                getTensorInfo<int32_t, index_type>(vi),
                getTensorInfo<int32_t, index_type>(index_img),
                getTensorInfo<scalar_t, index_type>(depth_img),
                getTensorInfo<scalar_t, index_type>(bary_img));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return {depth_img, bary_img};
}

torch::Tensor render_cuda_backward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& grad_depth_img,
    const torch::Tensor& grad_bary_img) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(v));

  auto N = v.size(0);
  auto V = v.size(1);
  auto C = v.size(2);
  auto H = index_img.size(1);
  auto W = index_img.size(2);
  int64_t count = N * H * W;

  auto grad_v = at::zeros({N, V, C}, v.options());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "interpolate_kernel", [&] {
      if (at::native::canUse32BitIndexMath(v) && at::native::canUse32BitIndexMath(grad_bary_img) &&
          at::native::canUse32BitIndexMath(grad_v) && at::native::canUse32BitIndexMath(index_img) &&
          at::native::canUse32BitIndexMath(grad_depth_img) &&
          at::native::canUse32BitIndexMath(vi)) {
        typedef int index_type;

        render_backward_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(count),
                getTensorInfo<scalar_t, index_type>(v),
                getTensorInfo<int32_t, index_type>(vi),
                getTensorInfo<int32_t, index_type>(index_img),
                getTensorInfo<scalar_t, index_type>(grad_depth_img),
                getTensorInfo<scalar_t, index_type>(grad_bary_img),
                getTensorInfo<scalar_t, index_type>(grad_v),
                grad_v.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;

        render_backward_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(count),
                getTensorInfo<scalar_t, index_type>(v),
                getTensorInfo<int32_t, index_type>(vi),
                getTensorInfo<int32_t, index_type>(index_img),
                getTensorInfo<scalar_t, index_type>(grad_depth_img),
                getTensorInfo<scalar_t, index_type>(grad_bary_img),
                getTensorInfo<scalar_t, index_type>(grad_v),
                grad_v.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return grad_v;
}

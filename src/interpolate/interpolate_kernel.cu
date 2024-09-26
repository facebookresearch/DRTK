// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <torch/types.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <cub/cub.cuh>

#include <kernel_utils.h>

using at::native::fastAtomicAdd;

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void interpolate_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> vert_attributes,
    TensorInfo<int32_t, index_t> vi,
    TensorInfo<int32_t, index_t> index_img,
    TensorInfo<scalar_t, index_t> bary_img,
    TensorInfo<scalar_t, index_t> out_img) {
  const index_t C = vert_attributes.sizes[2];
  const index_t H = bary_img.sizes[2];
  const index_t W = bary_img.sizes[3];

  const index_t vert_attributes_sN = vert_attributes.strides[0];
  const index_t vert_attributes_sV = vert_attributes.strides[1];
  const index_t vert_attributes_sC = vert_attributes.strides[2];

  const index_t vi_sV = vi.strides[0];
  const index_t vi_sF = vi.strides[1];

  const index_t index_img_sN = index_img.strides[0];
  const index_t index_img_sH = index_img.strides[1];
  const index_t index_img_sW = index_img.strides[2];

  const index_t bary_img_sN = bary_img.strides[0];
  const index_t bary_img_sB = bary_img.strides[1];
  const index_t bary_img_sH = bary_img.strides[2];
  const index_t bary_img_sW = bary_img.strides[3];

  const index_t out_img_sN = out_img.strides[0];
  const index_t out_img_sC = out_img.strides[1];
  const index_t out_img_sH = out_img.strides[2];
  const index_t out_img_sW = out_img.strides[3];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t n = index / (H * W);

    const int32_t tr_index = index_img.data[n * index_img_sN + h * index_img_sH + w * index_img_sW];
    scalar_t* __restrict out_ptr = out_img.data + out_img_sN * n + out_img_sH * h + out_img_sW * w;

    if (tr_index != -1) {
      const int32_t* __restrict vi_ptr = vi.data + tr_index * vi_sV;
      const int32_t vi_0 = vi_ptr[0 * vi_sF];
      const int32_t vi_1 = vi_ptr[1 * vi_sF];
      const int32_t vi_2 = vi_ptr[2 * vi_sF];

      const scalar_t* __restrict vert_ptr = vert_attributes.data + vert_attributes_sN * n;
      const scalar_t* vert_0_ptr = vert_ptr + vert_attributes_sV * vi_0;
      const scalar_t* vert_1_ptr = vert_ptr + vert_attributes_sV * vi_1;
      const scalar_t* vert_2_ptr = vert_ptr + vert_attributes_sV * vi_2;

      const scalar_t* __restrict bary_ptr =
          bary_img.data + bary_img_sN * n + bary_img_sH * h + bary_img_sW * w;
      const scalar_t bary_0 = bary_ptr[0 * bary_img_sB];
      const scalar_t bary_1 = bary_ptr[1 * bary_img_sB];
      const scalar_t bary_2 = bary_ptr[2 * bary_img_sB];

      for (int i = 0; i < C; ++i) {
        scalar_t v0 = vert_0_ptr[i * vert_attributes_sC];
        scalar_t v1 = vert_1_ptr[i * vert_attributes_sC];
        scalar_t v2 = vert_2_ptr[i * vert_attributes_sC];
        out_ptr[out_img_sC * i] = v0 * bary_0 + v1 * bary_1 + v2 * bary_2;
      }
    } else {
      for (int i = 0; i < C; ++i) {
        const scalar_t v[2] = {(w * 2.0f + 1.0f) / W - 1.0f, (h * 2.0f + 1.0f) / H - 1.0f};
        out_ptr[out_img_sC * i] = v[i % 2];
      }
    }
  }
}

template <typename scalar_t, typename index_t, bool bary_img_requires_grad, bool vert_requires_grad>
C10_LAUNCH_BOUNDS_1(256)
__global__ void interpolate_backward_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> grad_out,
    TensorInfo<scalar_t, index_t> vert_attributes,
    TensorInfo<int32_t, index_t> vi,
    TensorInfo<int32_t, index_t> index_img,
    TensorInfo<scalar_t, index_t> bary_img,
    TensorInfo<scalar_t, index_t> vert_attributes_grad,
    TensorInfo<scalar_t, index_t> bary_img_grad,
    const index_t memory_span) {
  index_t C = vert_attributes.sizes[2];
  index_t H = bary_img.sizes[2];
  index_t W = bary_img.sizes[3];

  index_t vert_attributes_sN = vert_attributes.strides[0];
  index_t vert_attributes_sV = vert_attributes.strides[1];
  index_t vert_attributes_sC = vert_attributes.strides[2];

  index_t vert_attributes_grad_sN = vert_attributes_grad.strides[0];
  index_t vert_attributes_grad_sV = vert_attributes_grad.strides[1];
  index_t vert_attributes_grad_sC = vert_attributes_grad.strides[2];

  index_t vi_sV = vi.strides[0];
  index_t vi_sF = vi.strides[1];

  index_t index_img_sN = index_img.strides[0];
  index_t index_img_sH = index_img.strides[1];
  index_t index_img_sW = index_img.strides[2];

  index_t bary_img_sN = bary_img.strides[0];
  index_t bary_img_sB = bary_img.strides[1];
  index_t bary_img_sH = bary_img.strides[2];
  index_t bary_img_sW = bary_img.strides[3];

  index_t bary_img_grad_sN = bary_img_grad.strides[0];
  index_t bary_img_grad_sB = bary_img_grad.strides[1];
  index_t bary_img_grad_sH = bary_img_grad.strides[2];
  index_t bary_img_grad_sW = bary_img_grad.strides[3];

  index_t grad_out_sN = grad_out.strides[0];
  index_t grad_out_sC = grad_out.strides[1];
  index_t grad_out_sH = grad_out.strides[2];
  index_t grad_out_sW = grad_out.strides[3];

  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int warp_size = 32;
  int lane = threadIdx.x % warp_size;

  __shared__ typename cub::WarpReduce<scalar_t>::TempStorage temp_storage_0;
  __shared__ typename cub::WarpReduce<scalar_t>::TempStorage temp_storage_1;
  __shared__ typename cub::WarpReduce<scalar_t>::TempStorage temp_storage_2;

  {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t n = index / (H * W);

    int32_t tr_index = -1;

    if (index < nthreads)
      tr_index = index_img.data[n * index_img_sN + h * index_img_sH + w * index_img_sW];
    const scalar_t* __restrict grad_out_ptr =
        grad_out.data + grad_out_sN * n + grad_out_sH * h + grad_out_sW * w;
    scalar_t* __restrict bary_grad_ptr =
        bary_img_grad.data + bary_img_grad_sN * n + bary_img_grad_sH * h + bary_img_grad_sW * w;

    bool thread_is_used = tr_index != -1;

    // True if at least one thread in the warp is used.
    bool warp_is_used = __any_sync(0xFFFFFFFFU, thread_is_used);

    if (warp_is_used) {
      int32_t vi_0 = -1, vi_1 = -1, vi_2 = -1;
      if (thread_is_used) {
        vi_0 = vi.data[tr_index * vi_sV + 0 * vi_sF];
        vi_1 = vi.data[tr_index * vi_sV + 1 * vi_sF];
        vi_2 = vi.data[tr_index * vi_sV + 2 * vi_sF];
      }
      unsigned m = 0xFFFFFFFFU;
      int vi_0_head = (__shfl_up_sync(m, vi_0, 1) != vi_0) || (lane == 0);
      int vi_0_tail = (__shfl_down_sync(m, vi_0, 1) != vi_0) || (lane == (warp_size - 1));
      int vi_1_head = (__shfl_up_sync(m, vi_1, 1) != vi_1) || (lane == 0);
      int vi_1_tail = (__shfl_down_sync(m, vi_1, 1) != vi_1) || (lane == (warp_size - 1));
      int vi_2_head = (__shfl_up_sync(m, vi_2, 1) != vi_2) || (lane == 0);
      int vi_2_tail = (__shfl_down_sync(m, vi_2, 1) != vi_2) || (lane == (warp_size - 1));

      const scalar_t* __restrict vert_ptr = vert_attributes.data + vert_attributes_sN * n;
      const scalar_t* vert_0_ptr = vert_ptr + vert_attributes_sV * vi_0;
      const scalar_t* vert_1_ptr = vert_ptr + vert_attributes_sV * vi_1;
      const scalar_t* vert_2_ptr = vert_ptr + vert_attributes_sV * vi_2;

      scalar_t* __restrict vert_grad_ptr = vert_attributes_grad.data + vert_attributes_grad_sN * n;
      scalar_t* vert_0_grad_ptr = vert_grad_ptr + vert_attributes_grad_sV * vi_0;
      scalar_t* vert_1_grad_ptr = vert_grad_ptr + vert_attributes_grad_sV * vi_1;
      scalar_t* vert_2_grad_ptr = vert_grad_ptr + vert_attributes_grad_sV * vi_2;

      const scalar_t* __restrict bary_ptr =
          bary_img.data + bary_img_sN * n + bary_img_sH * h + bary_img_sW * w;
      scalar_t bary_0, bary_1, bary_2;

      if (thread_is_used && vert_requires_grad) {
        bary_0 = bary_ptr[0 * bary_img_sB];
        bary_1 = bary_ptr[1 * bary_img_sB];
        bary_2 = bary_ptr[2 * bary_img_sB];
      }

      auto bary_0_grad = scalar_t(0.);
      auto bary_1_grad = scalar_t(0.);
      auto bary_2_grad = scalar_t(0.);

      for (int i = 0; i < C; ++i) {
        scalar_t g_out = grad_out_ptr[i * grad_out_sC];
        if (thread_is_used && bary_img_requires_grad) {
          scalar_t v0 = vert_0_ptr[i * vert_attributes_sC];
          scalar_t v1 = vert_1_ptr[i * vert_attributes_sC];
          scalar_t v2 = vert_2_ptr[i * vert_attributes_sC];

          bary_0_grad += g_out * v0;
          bary_1_grad += g_out * v1;
          bary_2_grad += g_out * v2;
        }

        if (vert_requires_grad) {
          scalar_t grad_v_0 =
              cub::WarpReduce<scalar_t>(temp_storage_0).TailSegmentedSum(g_out * bary_0, vi_0_tail);
          scalar_t grad_v_1 =
              cub::WarpReduce<scalar_t>(temp_storage_1).TailSegmentedSum(g_out * bary_1, vi_1_tail);
          scalar_t grad_v_2 =
              cub::WarpReduce<scalar_t>(temp_storage_2).TailSegmentedSum(g_out * bary_2, vi_2_tail);

          __syncthreads();

          if (vi_0_head && thread_is_used)
            fastAtomicAdd(
                vert_0_grad_ptr, i * vert_attributes_grad_sC, memory_span, grad_v_0, true);
          if (vi_1_head && thread_is_used)
            fastAtomicAdd(
                vert_1_grad_ptr, i * vert_attributes_grad_sC, memory_span, grad_v_1, true);
          if (vi_2_head && thread_is_used)
            fastAtomicAdd(
                vert_2_grad_ptr, i * vert_attributes_grad_sC, memory_span, grad_v_2, true);
        }
      }
      if (bary_img_requires_grad) {
        if (thread_is_used) {
          bary_grad_ptr[0 * bary_img_grad_sB] = bary_0_grad;
          bary_grad_ptr[1 * bary_img_grad_sB] = bary_1_grad;
          bary_grad_ptr[2 * bary_img_grad_sB] = bary_2_grad;
        } else {
          bary_grad_ptr[0 * bary_img_grad_sB] = scalar_t(0.);
          bary_grad_ptr[1 * bary_img_grad_sB] = scalar_t(0.);
          bary_grad_ptr[2 * bary_img_grad_sB] = scalar_t(0.);
        }
      }
    } else if ((index < nthreads) && bary_img_requires_grad) {
      bary_grad_ptr[0 * bary_img_grad_sB] = scalar_t(0.);
      bary_grad_ptr[1 * bary_img_grad_sB] = scalar_t(0.);
      bary_grad_ptr[2 * bary_img_grad_sB] = scalar_t(0.);
    }
  }
}

torch::Tensor interpolate_cuda(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  TORCH_CHECK(
      vert_attributes.defined() && vi.defined() && index_img.defined() && bary_img.defined(),
      "interpolate(): expected all inputs to be defined");
  auto vert_attributes_opt = vert_attributes.options();
  auto vi_opt = vi.options();
  auto index_img_opt = index_img.options();
  auto bary_img_opt = bary_img.options();
  TORCH_CHECK(
      (vert_attributes.device() == vi.device()) &&
          (vert_attributes.device() == index_img.device()) &&
          (vert_attributes.device() == bary_img.device()),
      "interpolate(): expected all inputs to be on same device");
  TORCH_CHECK(
      vert_attributes.dtype() == bary_img.dtype(),
      "interpolate(): expected vert_attributes and bary_img to have same dtype, but vert_attributes has ",
      vert_attributes.dtype(),
      " and bary_img has ",
      bary_img.dtype());
  TORCH_CHECK(
      vert_attributes.is_floating_point(),
      "interpolate(): expected vert_attributes to have floating point type, but v has ",
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
      (vert_attributes.dim() == 3) && (vi.dim() == 2) && (index_img.dim() == 3) &&
          (bary_img.dim() == 4),
      "interpolate(): expected vert_attributes.ndim == 3, vi.ndim == 2, index_img.ndim == 3, bary_img.ndim == 4, "
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
      "interpolate(): expected vert_attributes, index_img and bary_img to have same batch size, "
      "but got vert_attributes with sizes ",
      vert_attributes.sizes(),
      " and index_img with sizes ",
      index_img.sizes(),
      " and bary_img with sizes ",
      bary_img.sizes());
  TORCH_CHECK(
      vi.size(1) == 3 && bary_img.size(1) == 3,
      "interpolate(): expected second dim of vi to be of size 3, and second dim of bary_img to be of size 3, but got ",
      vi.size(1),
      " in the second dim of vi, and ",
      bary_img.size(1),
      " in the second dim of bary_img");
  TORCH_CHECK(
      index_img.size(1) == bary_img.size(2) && index_img.size(2) == bary_img.size(3),
      "interpolate(): expected H and W dims of index_img and bary_img to match");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(vert_attributes));

  auto N = vert_attributes.size(0);
  auto V = vert_attributes.size(1);
  auto C = vert_attributes.size(2);
  auto H = bary_img.size(2);
  auto W = bary_img.size(3);
  int64_t count = N * H * W;

  auto output = at::empty({N, C, H, W}, vert_attributes.options());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(vert_attributes.scalar_type(), "interpolate_kernel", [&] {
      if (at::native::canUse32BitIndexMath(vert_attributes) &&
          at::native::canUse32BitIndexMath(bary_img) &&
          at::native::canUse32BitIndexMath(index_img) && at::native::canUse32BitIndexMath(vi)) {
        typedef int index_type;

        interpolate_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(count),
                getTensorInfo<scalar_t, index_type>(vert_attributes),
                getTensorInfo<int32_t, index_type>(vi),
                getTensorInfo<int32_t, index_type>(index_img),
                getTensorInfo<scalar_t, index_type>(bary_img),
                getTensorInfo<scalar_t, index_type>(output));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;

        interpolate_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(count),
                getTensorInfo<scalar_t, index_type>(vert_attributes),
                getTensorInfo<int32_t, index_type>(vi),
                getTensorInfo<int32_t, index_type>(index_img),
                getTensorInfo<scalar_t, index_type>(bary_img),
                getTensorInfo<scalar_t, index_type>(output));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return output;
}

template <typename scalar_t, typename index_t, bool bary_img_requires_grad, bool vert_requires_grad>
void _interpolate_cuda_backward(
    int64_t count,
    const torch::Tensor& grad_out,
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    const torch::Tensor& vert_attributes_grad,
    const torch::Tensor& bary_img_grad) {
  interpolate_backward_kernel<scalar_t, index_t, bary_img_requires_grad, vert_requires_grad>
      <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<index_t>(count),
          getTensorInfo<scalar_t, index_t>(grad_out),
          getTensorInfo<scalar_t, index_t>(vert_attributes),
          getTensorInfo<int32_t, index_t>(vi),
          getTensorInfo<int32_t, index_t>(index_img),
          getTensorInfo<scalar_t, index_t>(bary_img),
          vert_requires_grad ? getTensorInfo<scalar_t, index_t>(vert_attributes_grad)
                             : TensorInfo<scalar_t, index_t>({nullptr, {0}, {0}, 0}),
          bary_img_requires_grad ? getTensorInfo<scalar_t, index_t>(bary_img_grad)
                                 : TensorInfo<scalar_t, index_t>({nullptr, {0}, {0}, 0}),
          vert_attributes_grad.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename index_t>
void _interpolate_cuda_backward(
    int64_t count,
    const torch::Tensor& grad_out,
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    const torch::Tensor& vert_attributes_grad,
    const torch::Tensor& bary_img_grad,
    bool bary_img_requires_grad,
    bool vert_requires_grad) {
  if (bary_img_requires_grad && vert_requires_grad)
    _interpolate_cuda_backward<scalar_t, index_t, true, true>(
        count,
        grad_out,
        vert_attributes,
        vi,
        index_img,
        bary_img,
        vert_attributes_grad,
        bary_img_grad);
  else if (bary_img_requires_grad)
    _interpolate_cuda_backward<scalar_t, index_t, true, false>(
        count,
        grad_out,
        vert_attributes,
        vi,
        index_img,
        bary_img,
        vert_attributes_grad,
        bary_img_grad);
  else if (vert_requires_grad)
    _interpolate_cuda_backward<scalar_t, index_t, false, true>(
        count,
        grad_out,
        vert_attributes,
        vi,
        index_img,
        bary_img,
        vert_attributes_grad,
        bary_img_grad);
}

std::tuple<torch::Tensor, torch::Tensor> interpolate_cuda_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vert_attributes));

  auto N = vert_attributes.size(0);
  auto V = vert_attributes.size(1);
  auto C = vert_attributes.size(2);
  auto H = bary_img.size(2);
  auto W = bary_img.size(3);
  int64_t count = N * H * W;

  bool bary_img_requires_grad = bary_img.requires_grad();
  bool vert_requires_grad = vert_attributes.requires_grad();

  auto vert_attributes_grad =
      vert_requires_grad ? at::zeros({N, V, C}, vert_attributes.options()) : torch::Tensor();
  auto bary_img_grad =
      bary_img_requires_grad ? at::empty({N, 3, H, W}, bary_img.options()) : torch::Tensor();

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(vert_attributes.scalar_type(), "interpolate_kernel", [&] {
      if (at::native::canUse32BitIndexMath(vert_attributes) &&
          at::native::canUse32BitIndexMath(bary_img) &&
          at::native::canUse32BitIndexMath(index_img) && at::native::canUse32BitIndexMath(vi)) {
        _interpolate_cuda_backward<scalar_t, int>(
            count,
            grad_out,
            vert_attributes,
            vi,
            index_img,
            bary_img,
            vert_attributes_grad,
            bary_img_grad,
            bary_img_requires_grad,
            vert_requires_grad);
      } else {
        _interpolate_cuda_backward<scalar_t, int64_t>(
            count,
            grad_out,
            vert_attributes,
            vi,
            index_img,
            bary_img,
            vert_attributes_grad,
            bary_img_grad,
            bary_img_requires_grad,
            vert_requires_grad);
      }
    });
  }
  return std::make_tuple(vert_attributes_grad, bary_img_grad);
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <torch/types.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <cub/cub.cuh>

#include <kernel_utils.h>

using at::native::fastAtomicAdd;

__device__ inline void sorted_corner_order_cuda(const int32_t cols[3], int order[3]) {
  order[0] = 0;
  order[1] = 1;
  order[2] = 2;
  if (cols[order[1]] < cols[order[0]]) {
    const int tmp = order[0];
    order[0] = order[1];
    order[1] = tmp;
  }
  if (cols[order[2]] < cols[order[1]]) {
    const int tmp = order[1];
    order[1] = order[2];
    order[2] = tmp;
  }
  if (cols[order[1]] < cols[order[0]]) {
    const int tmp = order[0];
    order[0] = order[1];
    order[1] = tmp;
  }
}

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

  const index_t vi_sN = vi.strides[0];
  const index_t vi_sV = vi.strides[1];
  const index_t vi_sF = vi.strides[2];

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
      const int32_t* __restrict vi_ptr = vi.data + n * vi_sN + tr_index * vi_sV;
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

  index_t vi_sN = vi.strides[0];
  index_t vi_sV = vi.strides[1];
  index_t vi_sF = vi.strides[2];

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

  constexpr int block_size = 256;
#ifdef __HIP_PLATFORM_AMD__
#ifdef __AMDGCN_WAVEFRONT_SIZE
  constexpr int warp_size = __AMDGCN_WAVEFRONT_SIZE;
#else
  constexpr int warp_size = 64;
#endif
  using WarpMask = uint64_t;
  constexpr WarpMask warp_mask = ~WarpMask{0};
#else
  constexpr int warp_size = 32;
  using WarpMask = unsigned;
  constexpr WarpMask warp_mask = 0xFFFFFFFFU;
#endif
  // Keep the mask, shuffle width, and CUB logical warp size aligned. AMD CDNA
  // wavefronts are 64 lanes, so treating them as 32-lane warps aliases lanes.
  static_assert(block_size % warp_size == 0, "block_size must be divisible by warp_size");
  constexpr int warps_per_block = block_size / warp_size;
  using WarpReduce = cub::WarpReduce<scalar_t, warp_size>;

  int warp_id = threadIdx.x / warp_size;
  int lane = threadIdx.x % warp_size;

  __shared__ typename WarpReduce::TempStorage temp_storage_0[warps_per_block];
  __shared__ typename WarpReduce::TempStorage temp_storage_1[warps_per_block];
  __shared__ typename WarpReduce::TempStorage temp_storage_2[warps_per_block];

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
    bool warp_is_used = __any_sync(warp_mask, thread_is_used);

    if (warp_is_used) {
      int32_t vi_0 = -1, vi_1 = -1, vi_2 = -1;
      if (thread_is_used) {
        const int32_t* __restrict vi_ptr = vi.data + n * vi_sN + tr_index * vi_sV;
        vi_0 = vi_ptr[0 * vi_sF];
        vi_1 = vi_ptr[1 * vi_sF];
        vi_2 = vi_ptr[2 * vi_sF];
      }
      int vi_0_head = (__shfl_up_sync(warp_mask, vi_0, 1, warp_size) != vi_0) || (lane == 0);
      int vi_0_tail =
          (__shfl_down_sync(warp_mask, vi_0, 1, warp_size) != vi_0) || (lane == (warp_size - 1));
      int vi_1_head = (__shfl_up_sync(warp_mask, vi_1, 1, warp_size) != vi_1) || (lane == 0);
      int vi_1_tail =
          (__shfl_down_sync(warp_mask, vi_1, 1, warp_size) != vi_1) || (lane == (warp_size - 1));
      int vi_2_head = (__shfl_up_sync(warp_mask, vi_2, 1, warp_size) != vi_2) || (lane == 0);
      int vi_2_tail =
          (__shfl_down_sync(warp_mask, vi_2, 1, warp_size) != vi_2) || (lane == (warp_size - 1));

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
              WarpReduce(temp_storage_0[warp_id]).TailSegmentedSum(g_out * bary_0, vi_0_tail);
          scalar_t grad_v_1 =
              WarpReduce(temp_storage_1[warp_id]).TailSegmentedSum(g_out * bary_1, vi_1_tail);
          scalar_t grad_v_2 =
              WarpReduce(temp_storage_2[warp_id]).TailSegmentedSum(g_out * bary_2, vi_2_tail);

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

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void interpolation_matrix_kernel(
    const int64_t nrows,
    const int32_t* __restrict__ vi,
    const int32_t* __restrict__ index_img,
    const scalar_t* __restrict__ bary_img,
    const int64_t* __restrict__ row_pixels,
    int64_t* __restrict__ col_indices,
    scalar_t* __restrict__ values,
    const int64_t F,
    const int64_t H,
    const int64_t W) {
  CUDA_KERNEL_LOOP_TYPE(row, nrows, int64_t) {
    const int64_t flat = row_pixels[row];
    const int64_t w = flat % W;
    const int64_t h = (flat / W) % H;
    const int64_t n = flat / (H * W);
    const int32_t tri = index_img[flat];

    const int32_t* __restrict__ face = vi + n * F * 3 + int64_t(tri) * 3;
    const int32_t cols[3] = {face[0], face[1], face[2]};
    const scalar_t* __restrict__ bary_pixel = bary_img + n * 3 * H * W + h * W + w;
    const scalar_t bary[3] = {
        bary_pixel[0 * H * W],
        bary_pixel[1 * H * W],
        bary_pixel[2 * H * W],
    };

    int order[3];
    sorted_corner_order_cuda(cols, order);
    for (int k = 0; k < 3; ++k) {
      const int corner = order[k];
      col_indices[row * 3 + k] = cols[corner];
      values[row * 3 + k] = bary[corner];
    }
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void interpolation_matrix_backward_kernel(
    const int64_t nrows,
    const scalar_t* __restrict__ grad_values,
    const int32_t* __restrict__ vi,
    const int32_t* __restrict__ index_img,
    const int64_t* __restrict__ row_pixels,
    scalar_t* __restrict__ bary_grad,
    const int64_t F,
    const int64_t H,
    const int64_t W) {
  CUDA_KERNEL_LOOP_TYPE(row, nrows, int64_t) {
    const int64_t flat = row_pixels[row];
    const int64_t w = flat % W;
    const int64_t h = (flat / W) % H;
    const int64_t n = flat / (H * W);
    const int32_t tri = index_img[flat];

    const int32_t* __restrict__ face = vi + n * F * 3 + int64_t(tri) * 3;
    const int32_t cols[3] = {face[0], face[1], face[2]};
    int order[3];
    sorted_corner_order_cuda(cols, order);

    scalar_t* __restrict__ bary_pixel = bary_grad + n * 3 * H * W + h * W + w;
    for (int k = 0; k < 3; ++k) {
      bary_pixel[order[k] * H * W] = grad_values[row * 3 + k];
    }
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void interpolation_normal_matrix_values_kernel(
    const int64_t nthreads,
    const int32_t* __restrict__ pair_indices,
    const int32_t* __restrict__ index_img,
    const scalar_t* __restrict__ bary_img,
    scalar_t* __restrict__ values,
    const int64_t F,
    const int64_t H,
    const int64_t W,
    const int64_t nnz) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, int64_t) {
    const int32_t tri = index_img[index];
    if (tri == -1) {
      continue;
    }

    const int64_t w = index % W;
    const int64_t h = (index / W) % H;
    const int64_t n = index / (H * W);
    const int32_t* __restrict__ pair = pair_indices + n * F * 9 + int64_t(tri) * 9;
    const scalar_t* __restrict__ bary_pixel = bary_img + n * 3 * H * W + h * W + w;
    const scalar_t b[3] = {
        bary_pixel[0 * H * W],
        bary_pixel[1 * H * W],
        bary_pixel[2 * H * W],
    };

    // pair_indices is built from searchsorted(unique_keys, keys), with
    // nnz == unique_keys.numel(), so every generated slot is in [0, nnz).
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        fastAtomicAdd(values, int64_t(pair[i * 3 + j]), nnz, b[i] * b[j], true);
      }
    }
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void interpolation_normal_matrix_values_backward_kernel(
    const int64_t nthreads,
    const scalar_t* __restrict__ grad_values,
    const int32_t* __restrict__ pair_indices,
    const int32_t* __restrict__ index_img,
    const scalar_t* __restrict__ bary_img,
    scalar_t* __restrict__ bary_grad,
    const int64_t F,
    const int64_t H,
    const int64_t W) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, int64_t) {
    const int32_t tri = index_img[index];
    if (tri == -1) {
      continue;
    }

    const int64_t w = index % W;
    const int64_t h = (index / W) % H;
    const int64_t n = index / (H * W);
    const int32_t* __restrict__ pair = pair_indices + n * F * 9 + int64_t(tri) * 9;
    const scalar_t* __restrict__ bary_pixel = bary_img + n * 3 * H * W + h * W + w;
    const scalar_t b0 = bary_pixel[0 * H * W];
    const scalar_t b1 = bary_pixel[1 * H * W];
    const scalar_t b2 = bary_pixel[2 * H * W];

    const scalar_t g00 = grad_values[pair[0]];
    const scalar_t g01 = grad_values[pair[1]];
    const scalar_t g02 = grad_values[pair[2]];
    const scalar_t g10 = grad_values[pair[3]];
    const scalar_t g11 = grad_values[pair[4]];
    const scalar_t g12 = grad_values[pair[5]];
    const scalar_t g20 = grad_values[pair[6]];
    const scalar_t g21 = grad_values[pair[7]];
    const scalar_t g22 = grad_values[pair[8]];

    scalar_t* __restrict__ grad_pixel = bary_grad + n * 3 * H * W + h * W + w;
    grad_pixel[0 * H * W] = scalar_t(2) * g00 * b0 + (g01 + g10) * b1 + (g02 + g20) * b2;
    grad_pixel[1 * H * W] = (g10 + g01) * b0 + scalar_t(2) * g11 * b1 + (g12 + g21) * b2;
    grad_pixel[2 * H * W] = (g20 + g02) * b0 + (g21 + g12) * b1 + scalar_t(2) * g22 * b2;
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
      "interpolate(): expected vert_attributes, index_img and bary_img to have same batch size, "
      "but got vert_attributes with sizes ",
      vert_attributes.sizes(),
      " and index_img with sizes ",
      index_img.sizes(),
      " and bary_img with sizes ",
      bary_img.sizes());
  TORCH_CHECK(
      vi.size(2) == 3 && bary_img.size(1) == 3,
      "interpolate(): expected last dim of vi to be of size 3, and second dim of bary_img to be of size 3, but got ",
      vi.size(2),
      " in the last dim of vi, and ",
      bary_img.size(1),
      " in the second dim of bary_img");
  TORCH_CHECK(
      vi.size(0) == vert_attributes.size(0),
      "interpolate(): expected vi to have same first dimension as vert_atrributes, but got ",
      vi.size(0),
      " in the first dim of vi, and ",
      vert_attributes.size(0),
      " in the first dim of vert_attributes");
  TORCH_CHECK(
      index_img.size(1) == bary_img.size(2) && index_img.size(2) == bary_img.size(3),
      "interpolate(): expected H and W dims of index_img and bary_img to match");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(vert_attributes));

  auto N = vert_attributes.size(0);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> interpolation_matrix_cuda(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  TORCH_CHECK(
      vi.defined() && index_img.defined() && bary_img.defined(),
      "interpolation_matrix(): expected all inputs to be defined");
  TORCH_CHECK(
      vi.device() == index_img.device() && vi.device() == bary_img.device(),
      "interpolation_matrix(): expected all inputs to be on same device");
  TORCH_CHECK(
      vi.dtype() == torch::kInt32,
      "interpolation_matrix(): expected vi to have int32 type, but vi has ",
      vi.dtype());
  TORCH_CHECK(
      index_img.dtype() == torch::kInt32,
      "interpolation_matrix(): expected index_img to have int32 type, but index_img has ",
      index_img.dtype());
  TORCH_CHECK(
      bary_img.is_floating_point(),
      "interpolation_matrix(): expected bary_img to have floating point type, but has ",
      bary_img.dtype());
  TORCH_CHECK(
      vi.dim() == 3 && index_img.dim() == 3 && bary_img.dim() == 4,
      "interpolation_matrix(): expected vi.ndim == 3, index_img.ndim == 3, bary_img.ndim == 4");
  TORCH_CHECK(
      vi.size(0) == index_img.size(0) && vi.size(0) == bary_img.size(0) && vi.size(2) == 3 &&
          bary_img.size(1) == 3 && index_img.size(1) == bary_img.size(2) &&
          index_img.size(2) == bary_img.size(3),
      "interpolation_matrix(): expected vi, index_img and bary_img shapes to agree");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(bary_img));

  auto vi_c = vi.contiguous();
  auto index_img_c = index_img.contiguous();
  auto bary_img_c = bary_img.contiguous();
  auto valid = index_img_c.reshape({-1}).ne(-1);
  auto row_pixels = at::nonzero(valid).reshape({-1});
  const int64_t num_rows = row_pixels.numel();
  auto long_options = index_img.options().dtype(torch::kInt64);
  auto crow_indices = at::arange(0, num_rows * 3 + 1, 3, long_options);
  auto col_indices = at::empty({num_rows * 3}, long_options);
  auto values = at::empty({num_rows * 3}, bary_img.options());

  if (num_rows > 0) {
    const int64_t F = vi_c.size(1);
    const int64_t H = index_img_c.size(1);
    const int64_t W = index_img_c.size(2);

    AT_DISPATCH_FLOATING_TYPES(bary_img_c.scalar_type(), "interpolation_matrix_cuda", [&] {
      interpolation_matrix_kernel<scalar_t>
          <<<GET_BLOCKS(num_rows, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
              num_rows,
              vi_c.data_ptr<int32_t>(),
              index_img_c.data_ptr<int32_t>(),
              bary_img_c.data_ptr<scalar_t>(),
              row_pixels.data_ptr<int64_t>(),
              col_indices.data_ptr<int64_t>(),
              values.data_ptr<scalar_t>(),
              F,
              H,
              W);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }

  return std::make_tuple(crow_indices, col_indices, values, row_pixels);
}

torch::Tensor interpolation_matrix_cuda_backward(
    const torch::Tensor& grad_values,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    const torch::Tensor& row_pixels) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(bary_img));

  auto vi_c = vi.contiguous();
  auto index_img_c = index_img.contiguous();
  auto bary_img_c = bary_img.contiguous();
  auto grad_values_c = grad_values.contiguous();
  auto row_pixels_c = row_pixels.contiguous();
  auto bary_grad = at::zeros_like(bary_img_c);

  const int64_t num_rows = row_pixels_c.numel();
  if (num_rows > 0) {
    const int64_t F = vi_c.size(1);
    const int64_t H = index_img_c.size(1);
    const int64_t W = index_img_c.size(2);

    AT_DISPATCH_FLOATING_TYPES(bary_img_c.scalar_type(), "interpolation_matrix_cuda_backward", [&] {
      interpolation_matrix_backward_kernel<scalar_t>
          <<<GET_BLOCKS(num_rows, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
              num_rows,
              grad_values_c.data_ptr<scalar_t>(),
              vi_c.data_ptr<int32_t>(),
              index_img_c.data_ptr<int32_t>(),
              row_pixels_c.data_ptr<int64_t>(),
              bary_grad.data_ptr<scalar_t>(),
              F,
              H,
              W);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }

  return bary_grad;
}

torch::Tensor interpolation_normal_matrix_values_cuda(
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t nnz) {
  TORCH_CHECK(
      pair_indices.defined() && index_img.defined() && bary_img.defined(),
      "interpolation_normal_matrix_values(): expected all inputs to be defined");
  TORCH_CHECK(
      pair_indices.device() == index_img.device() && pair_indices.device() == bary_img.device(),
      "interpolation_normal_matrix_values(): expected all inputs to be on same device");
  TORCH_CHECK(
      pair_indices.dtype() == torch::kInt32,
      "interpolation_normal_matrix_values(): expected pair_indices to have int32 type");
  TORCH_CHECK(
      index_img.dtype() == torch::kInt32,
      "interpolation_normal_matrix_values(): expected index_img to have int32 type");
  TORCH_CHECK(
      bary_img.is_floating_point(),
      "interpolation_normal_matrix_values(): expected bary_img to have floating point type");
  TORCH_CHECK(
      pair_indices.dim() == 3 && pair_indices.size(2) == 9 && index_img.dim() == 3 &&
          bary_img.dim() == 4 && bary_img.size(1) == 3,
      "interpolation_normal_matrix_values(): expected pair_indices [N,F,9], index_img [N,H,W], bary_img [N,3,H,W]");
  TORCH_CHECK(
      pair_indices.size(0) == index_img.size(0) && pair_indices.size(0) == bary_img.size(0) &&
          index_img.size(1) == bary_img.size(2) && index_img.size(2) == bary_img.size(3),
      "interpolation_normal_matrix_values(): expected pair_indices, index_img and bary_img shapes to agree");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(bary_img));

  auto pair_indices_c = pair_indices.contiguous();
  auto index_img_c = index_img.contiguous();
  auto bary_img_c = bary_img.contiguous();
  auto values = at::zeros({nnz}, bary_img.options());

  const int64_t count = index_img_c.numel();
  if (count > 0) {
    const int64_t F = pair_indices_c.size(1);
    const int64_t H = index_img_c.size(1);
    const int64_t W = index_img_c.size(2);

    AT_DISPATCH_FLOATING_TYPES(
        bary_img_c.scalar_type(), "interpolation_normal_matrix_values_cuda", [&] {
          interpolation_normal_matrix_values_kernel<scalar_t>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  count,
                  pair_indices_c.data_ptr<int32_t>(),
                  index_img_c.data_ptr<int32_t>(),
                  bary_img_c.data_ptr<scalar_t>(),
                  values.data_ptr<scalar_t>(),
                  F,
                  H,
                  W,
                  nnz);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  return values;
}

torch::Tensor interpolation_normal_matrix_values_cuda_backward(
    const torch::Tensor& grad_values,
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(bary_img));

  auto pair_indices_c = pair_indices.contiguous();
  auto index_img_c = index_img.contiguous();
  auto bary_img_c = bary_img.contiguous();
  auto grad_values_c = grad_values.contiguous();
  auto bary_grad = at::zeros_like(bary_img_c);

  const int64_t count = index_img_c.numel();
  if (count > 0) {
    const int64_t F = pair_indices_c.size(1);
    const int64_t H = index_img_c.size(1);
    const int64_t W = index_img_c.size(2);

    AT_DISPATCH_FLOATING_TYPES(
        bary_img_c.scalar_type(), "interpolation_normal_matrix_values_cuda_backward", [&] {
          interpolation_normal_matrix_values_backward_kernel<scalar_t>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  count,
                  grad_values_c.data_ptr<scalar_t>(),
                  pair_indices_c.data_ptr<int32_t>(),
                  index_img_c.data_ptr<int32_t>(),
                  bary_img_c.data_ptr<scalar_t>(),
                  bary_grad.data_ptr<scalar_t>(),
                  F,
                  H,
                  W);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  return bary_grad;
}

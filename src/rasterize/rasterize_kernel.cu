// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <grid_utils.h>
#include <torch/types.h>

#include <limits>

#include "rasterize_kernel.h"

#include <kernel_utils.h>

using namespace math;

template <typename scalar_t, typename index_t>
__global__ void rasterize_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> v,
    TensorInfo<int32_t, index_t> vi,
    TensorInfo<int64_t, index_t> packed_index_depth_img) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;
  typedef typename math::TVec4<scalar_t> scalar4_t;

  const index_t H = packed_index_depth_img.sizes[1];
  const index_t W = packed_index_depth_img.sizes[2];
  const index_t V = v.sizes[1];
  const index_t n_prim = vi.sizes[0];

  const index_t index_sN = packed_index_depth_img.strides[0];
  const index_t index_sH = packed_index_depth_img.strides[1];
  const index_t index_sW = packed_index_depth_img.strides[2];

  const index_t v_sN = v.strides[0];
  const index_t v_sV = v.strides[1];
  const index_t v_sC = v.strides[2];

  const index_t vi_sF = vi.strides[0];
  const index_t vi_sI = vi.strides[1];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t n = index / n_prim;
    const index_t id = index % n_prim;

    const int32_t* __restrict vi_ptr = vi.data + vi_sF * id;
    const int32_t vi_0 = (int32_t)(((uint32_t)vi_ptr[vi_sI * 0]) & 0x0FFFFFFFU);
    const int32_t vi_1 = vi_ptr[vi_sI * 1];
    const int32_t vi_2 = vi_ptr[vi_sI * 2];

    assert(vi_0 < V && vi_1 < V && vi_2 < V);

    const scalar_t* __restrict v_ptr = v.data + n * v_sN;
    const scalar2_t p_0 = {v_ptr[v_sV * vi_0 + v_sC * 0], v_ptr[v_sV * vi_0 + v_sC * 1]};
    const scalar2_t p_1 = {v_ptr[v_sV * vi_1 + v_sC * 0], v_ptr[v_sV * vi_1 + v_sC * 1]};
    const scalar2_t p_2 = {v_ptr[v_sV * vi_2 + v_sC * 0], v_ptr[v_sV * vi_2 + v_sC * 1]};

    const scalar3_t p_012_z = {
        v_ptr[v_sV * vi_0 + v_sC * 2],
        v_ptr[v_sV * vi_1 + v_sC * 2],
        v_ptr[v_sV * vi_2 + v_sC * 2]};

    const scalar2_t min_p = math::min(math::min(p_0, p_1), p_2);
    const scalar2_t max_p = math::max(math::max(p_0, p_1), p_2);

    const bool all_z_greater_0 = math::all_greater(p_012_z, {1e-8f, 1e-8f, 1e-8f});
    const bool in_canvas = math::all_less_or_eq(min_p, {(scalar_t)(W - 1), (scalar_t)(H - 1)}) &&
        math::all_greater(max_p, {0.f, 0.f});

    if (all_z_greater_0 && in_canvas) {
      const scalar2_t v_01 = p_1 - p_0;
      const scalar2_t v_02 = p_2 - p_0;
      const scalar2_t v_12 = p_2 - p_1;

      const scalar_t denominator = v_01.x * v_02.y - v_01.y * v_02.x;

      if (denominator != 0.f) {
        // Compute triangle bounds with extra border.
        int min_x = max(0, int(min_p.x));
        int min_y = max(0, int(min_p.y));

        int max_x = min((int)W - 1, int(max_p.x) + 1);
        int max_y = min((int)H - 1, int(max_p.y) + 1);

        // Loop over pixels inside triangle bbox.
        for (int y = min_y; y <= max_y; ++y) {
          for (int x = min_x; x <= max_x; ++x) {
            const scalar2_t p = {(scalar_t)x, (scalar_t)y};

            const scalar2_t vp0p = p - p_0;
            const scalar2_t vp1p = p - p_1;

            scalar3_t bary = scalar3_t({
                vp1p.y * v_12.x - vp1p.x * v_12.y,
                vp0p.x * v_02.y - vp0p.y * v_02.x,
                vp0p.y * v_01.x - vp0p.x * v_01.y,
            });
            bary *= sign(denominator);

            const bool on_edge_or_inside = (bary.x >= 0.f) && (bary.y >= 0.f) && (bary.z >= 0.f);

            bool on_edge_0 = bary.x == 0.f;
            bool on_edge_1 = bary.y == 0.f;
            bool on_edge_2 = bary.z == 0.f;

            const bool is_top_left_0 = (denominator > 0)
                ? (v_12.y < 0.f || v_12.y == 0.0f && v_12.x > 0.f)
                : (v_12.y > 0.f || v_12.y == 0.0f && v_12.x < 0.f);
            const bool is_top_left_1 = (denominator > 0)
                ? (v_02.y > 0.f || v_02.y == 0.0f && v_02.x < 0.f)
                : (v_02.y < 0.f || v_02.y == 0.0f && v_02.x > 0.f);
            const bool is_top_left_2 = (denominator > 0)
                ? (v_01.y < 0.f || v_01.y == 0.0f && v_01.x > 0.f)
                : (v_01.y > 0.f || v_01.y == 0.0f && v_01.x < 0.f);

            const bool is_top_left_or_inside = on_edge_or_inside &&
                !(on_edge_0 && !is_top_left_0 || on_edge_1 && !is_top_left_1 ||
                  on_edge_2 && !is_top_left_2);

            if (is_top_left_or_inside) {
              bary /= abs(denominator);

              // interpolate inverse depth linearly
              const scalar3_t d_inv = 1.0 / epsclamp(p_012_z);
              const scalar_t depth_inverse = dot(d_inv, bary);
              const scalar_t depth = 1.0f / epsclamp(depth_inverse);

              const unsigned long long packed_val =
                  (static_cast<unsigned long long>(__float_as_uint(depth)) << 32u) |
                  static_cast<unsigned long long>(id);
              atomicMin(
                  reinterpret_cast<unsigned long long*>(packed_index_depth_img.data) +
                      index_sN * n + index_sH * y + index_sW * x,
                  packed_val);
            }
          }
        }
      }
    }
  }
}

template <typename scalar_t>
__device__ inline void get_line(
    const math::TVec2<scalar_t>& p1,
    const math::TVec2<scalar_t>& p2,
    scalar_t& a,
    scalar_t& b,
    scalar_t& c) {
  a = p1.y - p2.y;
  b = p2.x - p1.x;
  c = p1.x * p2.y - p2.x * p1.y;
}

template <typename scalar_t>
__device__ inline bool is_point_in_segment(
    const math::TVec2<scalar_t>& p1,
    const math::TVec2<scalar_t>& p2,
    const math::TVec2<scalar_t>& c) {
  return (
      (((p2.x >= c.x) && (c.x >= p1.x)) || ((p2.x <= c.x) && (c.x <= p1.x))) &&
      (((p2.y >= c.y) && (c.y >= p1.y)) || ((p2.y <= c.y) && (c.y <= p1.y))));
}

template <typename scalar_t>
__device__ inline math::TVec2<scalar_t>
get_cross_point(scalar_t a1, scalar_t b1, scalar_t c1, scalar_t a2, scalar_t b2, scalar_t c2) {
  scalar_t d = a1 * b2 - a2 * b1;
  if (d == scalar_t(0)) {
    return math::TVec2<scalar_t>{std::numeric_limits<scalar_t>().max()};
  }
  return math::TVec2<scalar_t>{(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d};
}

template <typename scalar_t>
__device__ inline math::TVec2<scalar_t> get_cross_point(
    scalar_t a1,
    scalar_t b1,
    scalar_t c1,
    const math::TVec2<scalar_t>& p1,
    const math::TVec2<scalar_t>& p2) {
  scalar_t a2 = 1e16;
  scalar_t b2 = 1e16;
  scalar_t c2 = 1e16;
  get_line(p1, p2, a2, b2, c2);
  scalar_t d = a1 * b2 - a2 * b1;
  if (d == scalar_t(0)) {
    return math::TVec2<scalar_t>{std::numeric_limits<scalar_t>().max()};
  }
  return math::TVec2<scalar_t>{(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d};
}

template <typename scalar_t>
__device__ inline bool is_crossing_dimond(
    const math::TVec2<scalar_t>& p1,
    const math::TVec2<scalar_t>& p2,
    const math::TVec2<scalar_t>& p) {
  scalar_t a0 = 1e16;
  scalar_t b0 = 1e16;
  scalar_t c0 = 1e16;
  get_line(p1, p2, a0, b0, c0);
  bool intersecting = false;
  {
    math::TVec2<scalar_t> s0 = {p.x, p.y - scalar_t(0.5)};
    math::TVec2<scalar_t> s1 = {p.x + scalar_t(0.5), p.y};
    auto c = get_cross_point(a0, b0, c0, s0, s1);
    intersecting |=
        is_point_in_segment<scalar_t>(s0, s1, c) && is_point_in_segment<scalar_t>(p1, p2, c);
  }
  {
    math::TVec2<scalar_t> s0 = {p.x + scalar_t(0.5), p.y};
    math::TVec2<scalar_t> s1 = {p.x, p.y + scalar_t(0.5)};
    auto c = get_cross_point(a0, b0, c0, s0, s1);
    intersecting |=
        is_point_in_segment<scalar_t>(s0, s1, c) && is_point_in_segment<scalar_t>(p1, p2, c);
  }
  {
    math::TVec2<scalar_t> s0 = {p.x, p.y + scalar_t(0.5)};
    math::TVec2<scalar_t> s1 = {p.x - scalar_t(0.5), p.y};
    auto c = get_cross_point(a0, b0, c0, s0, s1);
    intersecting |=
        is_point_in_segment<scalar_t>(s0, s1, c) && is_point_in_segment<scalar_t>(p1, p2, c);
  }
  {
    math::TVec2<scalar_t> s0 = {p.x - scalar_t(0.5), p.y};
    math::TVec2<scalar_t> s1 = {p.x, p.y - scalar_t(0.5)};
    auto c = get_cross_point(a0, b0, c0, s0, s1);
    intersecting |=
        is_point_in_segment<scalar_t>(s0, s1, c) && is_point_in_segment<scalar_t>(p1, p2, c);
  }
  return intersecting;
}

template <typename scalar_t, typename index_t>
__global__ void rasterize_lines_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> v,
    TensorInfo<int32_t, index_t> vi,
    TensorInfo<int64_t, index_t> packed_index_depth_img) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;
  typedef typename math::TVec4<scalar_t> scalar4_t;

  const index_t H = packed_index_depth_img.sizes[1];
  const index_t W = packed_index_depth_img.sizes[2];
  const index_t V = v.sizes[1];
  const index_t n_prim = vi.sizes[0];

  const index_t index_sN = packed_index_depth_img.strides[0];
  const index_t index_sH = packed_index_depth_img.strides[1];
  const index_t index_sW = packed_index_depth_img.strides[2];

  const index_t v_sN = v.strides[0];
  const index_t v_sV = v.strides[1];
  const index_t v_sC = v.strides[2];

  const index_t vi_sF = vi.strides[0];
  const index_t vi_sI = vi.strides[1];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t n = index / n_prim;
    const index_t id = index % n_prim;

    const int32_t* __restrict vi_ptr = vi.data + vi_sF * id;
    const int32_t flag = (int32_t)((((uint32_t)vi_ptr[vi_sI * 0] & 0xF0000000U)) >> 28U);
    const int32_t vi_0 = (int32_t)(((uint32_t)vi_ptr[vi_sI * 0]) & 0x0FFFFFFFU);
    const int32_t vi_1 = vi_ptr[vi_sI * 1];
    const int32_t vi_2 = vi_ptr[vi_sI * 2];
    const bool edge_0_visible = (flag & 0b00000001) != 0;
    const bool edge_1_visible = (flag & 0b00000010) != 0;
    const bool edge_2_visible = (flag & 0b00000100) != 0;

    assert(vi_0 < V && vi_1 < V && vi_2 < V);

    const scalar_t* __restrict v_ptr = v.data + n * v_sN;
    const scalar2_t p_0 = {v_ptr[v_sV * vi_0 + v_sC * 0], v_ptr[v_sV * vi_0 + v_sC * 1]};
    const scalar2_t p_1 = {v_ptr[v_sV * vi_1 + v_sC * 0], v_ptr[v_sV * vi_1 + v_sC * 1]};
    const scalar2_t p_2 = {v_ptr[v_sV * vi_2 + v_sC * 0], v_ptr[v_sV * vi_2 + v_sC * 1]};

    const scalar3_t p_012_z = {
        v_ptr[v_sV * vi_0 + v_sC * 2],
        v_ptr[v_sV * vi_1 + v_sC * 2],
        v_ptr[v_sV * vi_2 + v_sC * 2]};

    const scalar2_t min_p = math::min(math::min(p_0, p_1), p_2);
    const scalar2_t max_p = math::max(math::max(p_0, p_1), p_2);

    const bool all_z_greater_0 = math::all_greater(p_012_z, {1e-8f, 1e-8f, 1e-8f});
    const bool in_canvas = math::all_less_or_eq(min_p, {(scalar_t)(W - 1), (scalar_t)(H - 1)}) &&
        math::all_greater(max_p, {0.f, 0.f});

    if (all_z_greater_0 && in_canvas) {
      const scalar2_t v_01 = p_1 - p_0;
      const scalar2_t v_02 = p_2 - p_0;
      const scalar2_t v_12 = p_2 - p_1;

      const scalar_t denominator = v_01.x * v_02.y - v_01.y * v_02.x;

      if (denominator != 0.f) {
        // Compute triangle bounds with extra border.
        int min_x = max(1, int(min_p.x) - 2);
        int min_y = max(1, int(min_p.y) - 2);

        int max_x = min((int)W - 2, int(max_p.x) + 2);
        int max_y = min((int)H - 2, int(max_p.y) + 2);

        // Loop over pixels inside triangle bbox.
        for (int y = min_y; y <= max_y; ++y) {
          for (int x = min_x; x <= max_x; ++x) {
            const scalar2_t p = {(scalar_t)x, (scalar_t)y};

            const scalar2_t vp0p = p - p_0;
            const scalar2_t vp1p = p - p_1;

            bool intersecting = false;
            intersecting |= is_crossing_dimond<scalar_t>(p_0, p_1, p) && edge_0_visible;
            intersecting |= is_crossing_dimond<scalar_t>(p_1, p_2, p) && edge_1_visible;
            intersecting |= is_crossing_dimond<scalar_t>(p_0, p_2, p) && edge_2_visible;

            scalar3_t bary = scalar3_t({
                vp1p.y * v_12.x - vp1p.x * v_12.y,
                vp0p.x * v_02.y - vp0p.y * v_02.x,
                vp0p.y * v_01.x - vp0p.x * v_01.y,
            });
            bary *= sign(denominator);

            const bool on_edge_or_inside = (bary.x >= 0.f) && (bary.y >= 0.f) && (bary.z >= 0.f);

            bool on_edge_0 = bary.x == 0.f;
            bool on_edge_1 = bary.y == 0.f;
            bool on_edge_2 = bary.z == 0.f;

            const bool is_top_left_0 = (denominator > 0)
                ? (v_12.y < 0.f || v_12.y == 0.0f && v_12.x > 0.f)
                : (v_12.y > 0.f || v_12.y == 0.0f && v_12.x < 0.f);
            const bool is_top_left_1 = (denominator > 0)
                ? (v_02.y > 0.f || v_02.y == 0.0f && v_02.x < 0.f)
                : (v_02.y < 0.f || v_02.y == 0.0f && v_02.x > 0.f);
            const bool is_top_left_2 = (denominator > 0)
                ? (v_01.y < 0.f || v_01.y == 0.0f && v_01.x > 0.f)
                : (v_01.y > 0.f || v_01.y == 0.0f && v_01.x < 0.f);

            const bool is_top_left_or_inside = on_edge_or_inside &&
                !(on_edge_0 && !is_top_left_0 || on_edge_1 && !is_top_left_1 ||
                  on_edge_2 && !is_top_left_2);

            if (is_top_left_or_inside || intersecting) {
              bary /= abs(denominator);
              bary = math::max(bary, scalar3_t{0, 0, 0});
              bary = math::min(bary, scalar3_t{1, 1, 1});
              bary = bary / math::sum(bary);

              // interpolate inverse depth linearly
              const scalar3_t d_inv = 1.0 / epsclamp(p_012_z);
              const scalar_t depth_inverse = dot(d_inv, bary);
              const scalar_t depth = 1.0f / epsclamp(depth_inverse);

              const unsigned long long packed_val =
                  (static_cast<unsigned long long>(__float_as_uint(depth)) << 32u) |
                  (intersecting ? static_cast<unsigned long long>(id) : 0xFFFFFFFFULL);
              atomicMin(
                  reinterpret_cast<unsigned long long*>(packed_index_depth_img.data) +
                      index_sN * n + index_sH * y + index_sW * x,
                  packed_val);
            }
          }
        }
      }
    }
  }
}

template <typename index_t>
__global__ void unpack_kernel(
    const index_t nthreads,
    TensorInfo<int64_t, index_t> packed_index_depth_img,
    TensorInfo<float, index_t> depth_img,
    TensorInfo<int32_t, index_t> index_img) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const unsigned long long int pv =
        reinterpret_cast<unsigned long long int*>(packed_index_depth_img.data)[index];
    const auto depth_uint = static_cast<uint32_t>(pv >> 32);
    depth_img.data[index] = depth_uint == 0xFFFFFFFF ? 0.0f : __uint_as_float(depth_uint);
    reinterpret_cast<uint32_t*>(index_img.data)[index] = static_cast<uint32_t>(pv & 0xFFFFFFFF);
  }
}

std::vector<torch::Tensor> rasterize_cuda(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    int64_t height,
    int64_t width,
    bool wireframe) {
  TORCH_CHECK(v.defined() && vi.defined(), "rasterize(): expected all inputs to be defined");
  auto v_opt = v.options();
  auto vi_opt = vi.options();
  TORCH_CHECK(
      (v.device() == vi.device()) && (v.is_cuda()),
      "rasterize(): expected all inputs to be on same cuda device");
  TORCH_CHECK(
      v.is_floating_point(),
      "rasterize(): expected v to have floating point type, but v has ",
      v.dtype());
  TORCH_CHECK(
      vi.dtype() == torch::kInt32,
      "rasterize(): expected vi to have int32 type, but vi has ",
      vi.dtype());
  TORCH_CHECK(
      v.layout() == torch::kStrided && vi.layout() == torch::kStrided,
      "rasterize(): expected all inputs to have torch.strided layout");
  TORCH_CHECK(
      (v.dim() == 3) && (vi.dim() == 2),
      "rasterize(): expected v.ndim == 3, vi.ndim == 2, "
      "but got v with sizes ",
      v.sizes(),
      " and vi with sizes ",
      vi.sizes());
  TORCH_CHECK(
      v.size(2) == 3 && vi.size(1) == 3,
      "rasterize(): expected third dim of v to be of size 3, and second dim of vi to be of size 3, but got ",
      v.size(2),
      " in the third dim of v, and ",
      vi.size(1),
      " in the second dim of vi");
  TORCH_CHECK(
      v.size(1) < 0x10000000U,
      "rasterize(): expected second dim of v to be less or eual to 268435456, but got ",
      v.size(1));
  TORCH_CHECK(
      height > 0 && width > 0,
      "rasterize(): both height and width have to be greater than zero, but got height: ",
      height,
      ", and width: ",
      width);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(v));
  auto stream = at::cuda::getCurrentCUDAStream();

  auto N = v.size(0);
  auto T = vi.size(0);
  auto H = height;
  auto W = width;
  const auto count_rasterize = N * T;
  const auto count_unpack = N * H * W;

  auto packed_index_depth_img = at::empty({N, H, W}, v.options().dtype(torch::kInt64));
  auto depth_img = at::empty({N, H, W}, v.options().dtype(torch::kFloat32));
  auto index_img = at::empty({N, H, W}, v.options().dtype(torch::kInt32));

  cudaMemsetAsync(
      packed_index_depth_img.data_ptr(),
      0xFF,
      N * H * W * torch::elementSize(torch::kInt64),
      stream);

  // rasterize
  if (count_rasterize > 0) {
    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "rasterize_kernel", [&] {
      if (at::native::canUse32BitIndexMath(v) && at::native::canUse32BitIndexMath(vi) &&
          at::native::canUse32BitIndexMath(packed_index_depth_img)) {
        typedef int index_type;

        if (wireframe) {
          rasterize_lines_kernel<scalar_t, index_type>
              <<<GET_BLOCKS(count_rasterize, 256), 256, 0, stream>>>(
                  static_cast<index_type>(count_rasterize),
                  getTensorInfo<scalar_t, index_type>(v),
                  getTensorInfo<int32_t, index_type>(vi),
                  getTensorInfo<int64_t, index_type>(packed_index_depth_img));
        } else {
          rasterize_kernel<scalar_t, index_type>
              <<<GET_BLOCKS(count_rasterize, 256), 256, 0, stream>>>(
                  static_cast<index_type>(count_rasterize),
                  getTensorInfo<scalar_t, index_type>(v),
                  getTensorInfo<int32_t, index_type>(vi),
                  getTensorInfo<int64_t, index_type>(packed_index_depth_img));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;

        if (wireframe) {
          rasterize_lines_kernel<scalar_t, index_type>
              <<<GET_BLOCKS(count_rasterize, 256), 256, 0, stream>>>(
                  static_cast<index_type>(count_rasterize),
                  getTensorInfo<scalar_t, index_type>(v),
                  getTensorInfo<int32_t, index_type>(vi),
                  getTensorInfo<int64_t, index_type>(packed_index_depth_img));
        } else {
          rasterize_kernel<scalar_t, index_type>
              <<<GET_BLOCKS(count_rasterize, 256), 256, 0, stream>>>(
                  static_cast<index_type>(count_rasterize),
                  getTensorInfo<scalar_t, index_type>(v),
                  getTensorInfo<int32_t, index_type>(vi),
                  getTensorInfo<int64_t, index_type>(packed_index_depth_img));
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }

  // unpack
  if (count_unpack > 0) {
    if (at::native::canUse32BitIndexMath(packed_index_depth_img) &&
        at::native::canUse32BitIndexMath(depth_img) &&
        at::native::canUse32BitIndexMath(index_img)) {
      typedef int index_type;

      unpack_kernel<index_type><<<GET_BLOCKS(count_rasterize, 256), 256, 0, stream>>>(
          static_cast<index_type>(count_unpack),
          getTensorInfo<int64_t, index_type>(packed_index_depth_img),
          getTensorInfo<float, index_type>(depth_img),
          getTensorInfo<int32_t, index_type>(index_img));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      typedef int64_t index_type;

      unpack_kernel<index_type><<<GET_BLOCKS(count_rasterize, 256), 256, 0, stream>>>(
          static_cast<index_type>(count_unpack),
          getTensorInfo<int64_t, index_type>(packed_index_depth_img),
          getTensorInfo<float, index_type>(depth_img),
          getTensorInfo<int32_t, index_type>(index_img));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  return {depth_img, index_img};
}

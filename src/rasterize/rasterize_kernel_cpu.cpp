// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Parallel.h>
#include <cpu_atomic.h>
#include <cuda_math_helper.h>
#include <torch/types.h>

#include <cstring>
#include <limits>

#include "rasterize_kernel.h"

using namespace math;

namespace {

inline uint32_t float_as_uint(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

inline float uint_as_float(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

template <typename scalar_t>
void rasterize_triangles_cpu(
    const scalar_t* v_ptr,
    const int32_t* vi_ptr,
    int64_t* packed_buf,
    int64_t N,
    int64_t V,
    int64_t n_prim,
    int64_t H,
    int64_t W,
    // Strides (in elements, not bytes)
    int64_t v_sN,
    int64_t v_sV,
    int64_t v_sC,
    int64_t vi_sN,
    int64_t vi_sF,
    int64_t vi_sI) {
  const int64_t total = N * n_prim;

  at::parallel_for(0, total, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t n = index / n_prim;
      const int64_t id = index % n_prim;

      const int32_t* vi_face = vi_ptr + vi_sN * n + vi_sF * id;
      const int32_t vi_0 =
          static_cast<int32_t>(static_cast<uint32_t>(vi_face[vi_sI * 0]) & 0x0FFFFFFFU);
      const int32_t vi_1 = vi_face[vi_sI * 1];
      const int32_t vi_2 = vi_face[vi_sI * 2];

      // Skip degenerate triangles
      if ((vi_0 == vi_1) && (vi_1 == vi_2))
        continue;

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

      // All z must be > 0
      if (!(p0_z > 1e-8f && p1_z > 1e-8f && p2_z > 1e-8f))
        continue;

      const scalar_t min_x = std::min({p0_x, p1_x, p2_x});
      const scalar_t min_y = std::min({p0_y, p1_y, p2_y});
      const scalar_t max_x = std::max({p0_x, p1_x, p2_x});
      const scalar_t max_y = std::max({p0_y, p1_y, p2_y});

      // Check if triangle is in canvas
      if (!(min_x <= static_cast<scalar_t>(W - 1) && min_y <= static_cast<scalar_t>(H - 1) &&
            max_x > 0.f && max_y > 0.f))
        continue;

      const scalar_t v01_x = p1_x - p0_x;
      const scalar_t v01_y = p1_y - p0_y;
      const scalar_t v02_x = p2_x - p0_x;
      const scalar_t v02_y = p2_y - p0_y;
      const scalar_t v12_x = p2_x - p1_x;
      const scalar_t v12_y = p2_y - p1_y;

      const scalar_t denominator = v01_x * v02_y - v01_y * v02_x;
      if (denominator == 0.f)
        continue;

      const scalar_t sign_denom = denominator > 0 ? scalar_t(1) : scalar_t(-1);
      const scalar_t abs_denom = std::abs(denominator);

      // Bounding box clamped to image
      const int bb_min_x = std::max(0, static_cast<int>(min_x));
      const int bb_min_y = std::max(0, static_cast<int>(min_y));
      const int bb_max_x = std::min(static_cast<int>(W) - 1, static_cast<int>(max_x) + 1);
      const int bb_max_y = std::min(static_cast<int>(H) - 1, static_cast<int>(max_y) + 1);

      // Inverse depth for the three vertices
      const scalar_t d_inv_0 = scalar_t(1) / epsclamp(p0_z);
      const scalar_t d_inv_1 = scalar_t(1) / epsclamp(p1_z);
      const scalar_t d_inv_2 = scalar_t(1) / epsclamp(p2_z);

      // Top-left rule edge classification
      const bool is_top_left_0 = (denominator > 0) ? (v12_y < 0.f || (v12_y == 0.f && v12_x > 0.f))
                                                   : (v12_y > 0.f || (v12_y == 0.f && v12_x < 0.f));
      const bool is_top_left_1 = (denominator > 0) ? (v02_y > 0.f || (v02_y == 0.f && v02_x < 0.f))
                                                   : (v02_y < 0.f || (v02_y == 0.f && v02_x > 0.f));
      const bool is_top_left_2 = (denominator > 0) ? (v01_y < 0.f || (v01_y == 0.f && v01_x > 0.f))
                                                   : (v01_y > 0.f || (v01_y == 0.f && v01_x < 0.f));

      for (int y = bb_min_y; y <= bb_max_y; ++y) {
        for (int x = bb_min_x; x <= bb_max_x; ++x) {
          const scalar_t px = static_cast<scalar_t>(x);
          const scalar_t py = static_cast<scalar_t>(y);

          const scalar_t vp0_x = px - p0_x;
          const scalar_t vp0_y = py - p0_y;
          const scalar_t vp1_x = px - p1_x;
          const scalar_t vp1_y = py - p1_y;

          scalar_t bary_0 = (vp1_y * v12_x - vp1_x * v12_y) * sign_denom;
          scalar_t bary_1 = (vp0_x * v02_y - vp0_y * v02_x) * sign_denom;
          scalar_t bary_2 = (vp0_y * v01_x - vp0_x * v01_y) * sign_denom;

          const bool on_edge_or_inside = (bary_0 >= 0.f) && (bary_1 >= 0.f) && (bary_2 >= 0.f);

          if (!on_edge_or_inside)
            continue;

          const bool on_edge_0 = bary_0 == 0.f;
          const bool on_edge_1 = bary_1 == 0.f;
          const bool on_edge_2 = bary_2 == 0.f;

          const bool is_top_left_or_inside = on_edge_or_inside &&
              !((!is_top_left_0 && on_edge_0) || (!is_top_left_1 && on_edge_1) ||
                (!is_top_left_2 && on_edge_2));

          if (!is_top_left_or_inside)
            continue;

          bary_0 /= abs_denom;
          bary_1 /= abs_denom;
          bary_2 /= abs_denom;

          // Interpolate inverse depth linearly
          const scalar_t depth_inverse = d_inv_0 * bary_0 + d_inv_1 * bary_1 + d_inv_2 * bary_2;
          const float depth = static_cast<float>(scalar_t(1) / epsclamp(depth_inverse));

          const uint64_t packed_val = (static_cast<uint64_t>(float_as_uint(depth)) << 32u) |
              static_cast<uint64_t>(static_cast<uint32_t>(id));

          drtk::atomic_min_unsigned(
              packed_buf + n * H * W + y * W + x, static_cast<int64_t>(packed_val));
        }
      }
    }
  });
}

void unpack_cpu(const int64_t* packed_ptr, float* depth_ptr, int32_t* index_ptr, int64_t count) {
  at::parallel_for(0, count, /*grain_size=*/4096, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const auto pv = static_cast<uint64_t>(packed_ptr[i]);
      const auto depth_uint = static_cast<uint32_t>(pv >> 32);
      depth_ptr[i] = depth_uint == 0xFFFFFFFF ? 0.0f : uint_as_float(depth_uint);
      index_ptr[i] = static_cast<int32_t>(static_cast<uint32_t>(pv & 0xFFFFFFFF));
    }
  });
}

} // namespace

std::vector<torch::Tensor> rasterize_cpu(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    int64_t height,
    int64_t width,
    bool wireframe) {
  TORCH_CHECK(v.defined() && vi.defined(), "rasterize(): expected all inputs to be defined");
  TORCH_CHECK(
      v.device().is_cpu() && vi.device().is_cpu(), "rasterize(): expected all inputs to be on CPU");
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
      (v.dim() == 3) && (vi.dim() == 3),
      "rasterize(): expected v.ndim == 3, vi.ndim == 3, "
      "but got v with sizes ",
      v.sizes(),
      " and vi with sizes ",
      vi.sizes());
  TORCH_CHECK(
      v.size(2) == 3 && vi.size(2) == 3,
      "rasterize(): expected third dim of v and last dim of vi to be 3, but got ",
      v.size(2),
      " and ",
      vi.size(2));
  TORCH_CHECK(
      vi.size(0) == v.size(0),
      "rasterize(): expected first dim of vi to match first dim of v, but got ",
      v.size(0),
      " and ",
      vi.size(0));
  TORCH_CHECK(
      v.size(1) < 0x10000000U,
      "rasterize(): expected second dim of v to be less than 268435456, but got ",
      v.size(1));
  TORCH_CHECK(
      height > 0 && width > 0,
      "rasterize(): both height and width must be > 0, but got height: ",
      height,
      ", width: ",
      width);
  TORCH_CHECK(!wireframe, "rasterize(): wireframe mode is not supported on CPU");

  // Contiguous copies for predictable stride arithmetic
  auto v_c = v.contiguous();
  auto vi_c = vi.contiguous();

  const auto N = v_c.size(0);
  const auto V = v_c.size(1);
  const auto T = vi_c.size(1);
  const auto H = height;
  const auto W = width;

  // Allocate packed buffer, depth image, and index image on CPU
  auto packed_buf = at::empty({N, H, W}, v.options().dtype(torch::kInt64));
  auto depth_img = at::empty({N, H, W}, v.options().dtype(torch::kFloat32));
  auto index_img = at::empty({N, H, W}, v.options().dtype(torch::kInt32));

  // Fill with 0xFF — same as CUDA path. This sets every packed value to the
  // maximum uint64, meaning "no triangle".
  std::memset(packed_buf.data_ptr(), 0xFF, N * H * W * sizeof(int64_t));

  if (N * T > 0) {
    AT_DISPATCH_FLOATING_TYPES(v_c.scalar_type(), "rasterize_cpu", [&] {
      rasterize_triangles_cpu<scalar_t>(
          v_c.data_ptr<scalar_t>(),
          vi_c.data_ptr<int32_t>(),
          packed_buf.data_ptr<int64_t>(),
          N,
          V,
          T,
          H,
          W,
          /*v_sN=*/V * 3,
          /*v_sV=*/3,
          /*v_sC=*/1,
          /*vi_sN=*/T * 3,
          /*vi_sF=*/3,
          /*vi_sI=*/1);
    });
  }

  const auto count = N * H * W;
  if (count > 0) {
    unpack_cpu(
        packed_buf.data_ptr<int64_t>(),
        depth_img.data_ptr<float>(),
        index_img.data_ptr<int32_t>(),
        count);
  }

  return {depth_img, index_img};
}

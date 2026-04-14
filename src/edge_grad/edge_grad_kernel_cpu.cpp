// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Parallel.h>
#include <cpu_atomic.h>
#include <cuda_math_helper.h>
#include <torch/types.h>

#include "edge_grad_kernel.h"

using namespace math;
using drtk::atomic_add;

namespace {

template <typename scalar_t>
struct TriInfo {
  typedef TVec2<scalar_t> scalar2_t;

  scalar2_t p_0;
  scalar2_t p_1;
  scalar2_t v_01;
  scalar2_t v_02;
  scalar2_t v_12;
  scalar_t denominator;
};

template <typename scalar_t>
bool pix_in_tri(const TriInfo<scalar_t>& tri, int x, int y) {
  typedef TVec2<scalar_t> scalar2_t;
  typedef TVec3<scalar_t> scalar3_t;

  if (tri.denominator != scalar_t(0)) {
    const scalar2_t p = {(scalar_t)x, (scalar_t)y};

    const scalar2_t vp0p = p - tri.p_0;
    const scalar2_t vp1p = p - tri.p_1;

    scalar3_t bary = scalar3_t({
        vp1p.y * tri.v_12.x - vp1p.x * tri.v_12.y,
        vp0p.x * tri.v_02.y - vp0p.y * tri.v_02.x,
        vp0p.y * tri.v_01.x - vp0p.x * tri.v_01.y,
    });
    bary *= sign(tri.denominator);

    const bool on_edge_or_inside =
        (bary.x >= scalar_t(0)) && (bary.y >= scalar_t(0)) && (bary.z >= scalar_t(0));

    bool on_edge_0 = bary.x == scalar_t(0);
    bool on_edge_1 = bary.y == scalar_t(0);
    bool on_edge_2 = bary.z == scalar_t(0);

    const bool is_top_left_0 = (tri.denominator > 0)
        ? (tri.v_12.y < scalar_t(0) || (tri.v_12.y == scalar_t(0) && tri.v_12.x > scalar_t(0)))
        : (tri.v_12.y > scalar_t(0) || (tri.v_12.y == scalar_t(0) && tri.v_12.x < scalar_t(0)));
    const bool is_top_left_1 = (tri.denominator > 0)
        ? (tri.v_02.y > scalar_t(0) || (tri.v_02.y == scalar_t(0) && tri.v_02.x < scalar_t(0)))
        : (tri.v_02.y < scalar_t(0) || (tri.v_02.y == scalar_t(0) && tri.v_02.x > scalar_t(0)));
    const bool is_top_left_2 = (tri.denominator > 0)
        ? (tri.v_01.y < scalar_t(0) || (tri.v_01.y == scalar_t(0) && tri.v_01.x > scalar_t(0)))
        : (tri.v_01.y > scalar_t(0) || (tri.v_01.y == scalar_t(0) && tri.v_01.x < scalar_t(0)));

    const bool is_top_left_or_inside = on_edge_or_inside &&
        !((on_edge_0 && !is_top_left_0) || (on_edge_1 && !is_top_left_1) ||
          (on_edge_2 && !is_top_left_2));
    return is_top_left_or_inside;
  }
  return false;
}

template <typename scalar_t>
TriInfo<scalar_t> get_tri_info(
    const scalar_t* v_ptr,
    int64_t v_sV,
    int64_t v_sC,
    int32_t vi0,
    int32_t vi1,
    int32_t vi2) {
  typedef TVec2<scalar_t> scalar2_t;
  const scalar2_t p_0 = {v_ptr[v_sV * vi0 + v_sC * 0], v_ptr[v_sV * vi0 + v_sC * 1]};
  const scalar2_t p_1 = {v_ptr[v_sV * vi1 + v_sC * 0], v_ptr[v_sV * vi1 + v_sC * 1]};
  const scalar2_t p_2 = {v_ptr[v_sV * vi2 + v_sC * 0], v_ptr[v_sV * vi2 + v_sC * 1]};

  const scalar2_t v_01 = p_1 - p_0;
  const scalar2_t v_02 = p_2 - p_0;
  const scalar2_t v_12 = p_2 - p_1;

  const scalar_t denominator = v_01.x * v_02.y - v_01.y * v_02.x;

  return {p_0, p_1, v_01, v_02, v_12, denominator};
}

template <typename scalar_t>
TVec3<scalar_t> get_tri_normal(
    const scalar_t* v_ptr,
    int64_t v_sV,
    int64_t v_sC,
    int32_t vi0,
    int32_t vi1,
    int32_t vi2) {
  typedef TVec3<scalar_t> scalar3_t;
  const scalar3_t p_0 = {
      v_ptr[v_sV * vi0 + v_sC * 0], v_ptr[v_sV * vi0 + v_sC * 1], v_ptr[v_sV * vi0 + v_sC * 2]};
  const scalar3_t p_1 = {
      v_ptr[v_sV * vi1 + v_sC * 0], v_ptr[v_sV * vi1 + v_sC * 1], v_ptr[v_sV * vi1 + v_sC * 2]};
  const scalar3_t p_2 = {
      v_ptr[v_sV * vi2 + v_sC * 0], v_ptr[v_sV * vi2 + v_sC * 1], v_ptr[v_sV * vi2 + v_sC * 2]};
  return normalize(cross(p_0 - p_2, p_1 - p_0));
}

// See edge_grad_kernel.cu get_dp_dr() for the full derivation and comment.
template <typename scalar_t>
TVec2<scalar_t> get_dp_dr(
    const TVec2<scalar_t>& n_varying_,
    const TVec2<scalar_t>& n_fixed_,
    scalar_t max_magnitude = scalar_t(0)) {
  typedef TVec2<scalar_t> scalar2_t;

  const auto n_varying = normalize(n_varying_);
  const auto n_fixed = normalize(n_fixed_);
  const scalar2_t b = {-n_fixed.y, n_fixed.x};
  const auto d = dot(b, n_varying);
  if (max_magnitude > scalar_t(0)) {
    // Clamp |b.x/d| ≤ M by ensuring |d| ≥ |b.x|/M.
    const auto abs_d = std::abs(d);
    const auto abs_bx_over_M = std::abs(b.x) / max_magnitude;
    // When both d and b.x are zero (n_fixed along first axis), max(0,0)=0.
    // epsclamp prevents 0/0; the result is 0 anyway since b.x=0.
    const auto safe_d =
        (d >= scalar_t(0) ? scalar_t(1) : scalar_t(-1)) * epsclamp(std::max(abs_d, abs_bx_over_M));
    return (b.x / safe_d) * n_varying;
  } else {
    return b.x / epsclamp(d) * n_varying;
  }
}

template <typename scalar_t>
void edge_grad_backward_cpu_impl(
    const scalar_t* v_pix_ptr,
    const scalar_t* img_ptr,
    const int32_t* index_img_ptr,
    const int32_t* vi_ptr,
    const scalar_t* grad_output_ptr,
    scalar_t* grad_v_pix_img_ptr,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    // v_pix strides
    int64_t vp_sN,
    int64_t vp_sV,
    int64_t vp_sC,
    // vi strides
    int64_t vi_sN,
    int64_t vi_sV,
    int64_t vi_sF,
    // index_img strides
    int64_t idx_sN,
    int64_t idx_sH,
    int64_t idx_sW,
    // img strides
    int64_t img_sN,
    int64_t img_sC,
    int64_t img_sH,
    int64_t img_sW,
    // grad_output strides
    int64_t go_sN,
    int64_t go_sC,
    int64_t go_sH,
    int64_t go_sW,
    // grad_v_pix_img strides
    int64_t gvpi_sN,
    int64_t gvpi_sC,
    int64_t gvpi_sH,
    int64_t gvpi_sW,
    double max_dp_dr) {
  typedef TVec2<scalar_t> scalar2_t;
  typedef TVec3<scalar_t> scalar3_t;

  const int64_t count = N * H * W;

  at::parallel_for(0, count, /*grain_size=*/4096, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t x = index % W;
      const int64_t y = (index / W) % H;
      const int64_t n = index / (H * W);

      if (x >= (W - 1) || y >= (H - 1))
        continue;

      // CRD (center-right-down) 2x2 neighborhood
      const int32_t* idx_n = index_img_ptr + n * idx_sN;
      const int32_t center_index = idx_n[(y + 0) * idx_sH + (x + 0) * idx_sW];
      const int32_t right_index = idx_n[(y + 0) * idx_sH + (x + 1) * idx_sW];
      const int32_t down_index = idx_n[(y + 1) * idx_sH + (x + 0) * idx_sW];

      const bool c_valid = (center_index >= 0);
      const bool r_valid = (right_index >= 0);
      const bool d_valid = (down_index >= 0);

      // Vertex indices for each triangle (0,0,0 if invalid)
      int32_t vi_c0 = 0, vi_c1 = 0, vi_c2 = 0;
      int32_t vi_r0 = 0, vi_r1 = 0, vi_r2 = 0;
      int32_t vi_d0 = 0, vi_d1 = 0, vi_d2 = 0;

      const int32_t* vi_n = vi_ptr + n * vi_sN;
      if (c_valid) {
        const int32_t* f = vi_n + center_index * vi_sV;
        vi_c0 = f[0 * vi_sF];
        vi_c1 = f[1 * vi_sF];
        vi_c2 = f[2 * vi_sF];
      }
      if (r_valid) {
        const int32_t* f = vi_n + right_index * vi_sV;
        vi_r0 = f[0 * vi_sF];
        vi_r1 = f[1 * vi_sF];
        vi_r2 = f[2 * vi_sF];
      }
      if (d_valid) {
        const int32_t* f = vi_n + down_index * vi_sV;
        vi_d0 = f[0 * vi_sF];
        vi_d1 = f[1 * vi_sF];
        vi_d2 = f[2 * vi_sF];
      }

      const bool lr_diff = (center_index != right_index);
      const bool ud_diff = (center_index != down_index);

      const bool x_both_valid = c_valid && r_valid;
      const bool y_both_valid = c_valid && d_valid;

      const scalar_t* v_pix_n = v_pix_ptr + n * vp_sN;
      const auto tri_center = get_tri_info(v_pix_n, vp_sV, vp_sC, vi_c0, vi_c1, vi_c2);
      const auto tri_right = get_tri_info(v_pix_n, vp_sV, vp_sC, vi_r0, vi_r1, vi_r2);
      const auto tri_down = get_tri_info(v_pix_n, vp_sV, vp_sC, vi_d0, vi_d1, vi_d2);

      const bool center_pix_in_right_tri =
          lr_diff && x_both_valid && pix_in_tri(tri_right, (int)x, (int)y);
      const bool right_pix_in_center_tri =
          lr_diff && x_both_valid && pix_in_tri(tri_center, (int)x + 1, (int)y);
      const bool center_pix_in_down_tri =
          ud_diff && y_both_valid && pix_in_tri(tri_down, (int)x, (int)y);
      const bool down_pix_in_center_tri =
          ud_diff && y_both_valid && pix_in_tri(tri_center, (int)x, (int)y + 1);

      const bool l_over_r = center_pix_in_right_tri && (!right_pix_in_center_tri);
      const bool r_over_l = right_pix_in_center_tri && (!center_pix_in_right_tri);
      const bool u_over_d = center_pix_in_down_tri && (!down_pix_in_center_tri);
      const bool d_over_u = down_pix_in_center_tri && (!center_pix_in_down_tri);

      const bool horiz_int = center_pix_in_right_tri && right_pix_in_center_tri;
      const bool vert_int = center_pix_in_down_tri && down_pix_in_center_tri;

      const bool horiz_adjacent =
          lr_diff && x_both_valid && (!center_pix_in_right_tri && !right_pix_in_center_tri);
      const bool vert_adjacent =
          ud_diff && y_both_valid && (!center_pix_in_down_tri && !down_pix_in_center_tri);

      // Compute image gradient dot output gradient
      const scalar_t* img_n = img_ptr + img_sN * n;
      const scalar_t* go_n = grad_output_ptr + go_sN * n;

      scalar_t grad_dot_x = scalar_t(0);
      scalar_t grad_dot_y = scalar_t(0);

      if (lr_diff) {
        const scalar_t* img_right = img_n + y * img_sH + (x + 1) * img_sW;
        const scalar_t* img_center = img_n + y * img_sH + (x + 0) * img_sW;
        const scalar_t* go_right = go_n + y * go_sH + (x + 1) * go_sW;
        const scalar_t* go_center = go_n + y * go_sH + (x + 0) * go_sW;
        for (int64_t c = 0; c < C; ++c) {
          grad_dot_x += (img_right[c * img_sC] - img_center[c * img_sC]) *
              (scalar_t(0.5) * (go_right[c * go_sC] + go_center[c * go_sC]));
        }
      }
      if (ud_diff) {
        const scalar_t* img_down = img_n + (y + 1) * img_sH + x * img_sW;
        const scalar_t* img_center = img_n + (y + 0) * img_sH + x * img_sW;
        const scalar_t* go_down = go_n + (y + 1) * go_sH + x * go_sW;
        const scalar_t* go_center = go_n + (y + 0) * go_sH + x * go_sW;
        for (int64_t c = 0; c < C; ++c) {
          grad_dot_y += (img_down[c * img_sC] - img_center[c * img_sC]) *
              (scalar_t(0.5) * (go_down[c * go_sC] + go_center[c * go_sC]));
        }
      }

      scalar3_t grad_v_pix_center = {scalar_t(0), scalar_t(0), scalar_t(0)};
      scalar3_t grad_v_pix_right = {scalar_t(0), scalar_t(0), scalar_t(0)};
      scalar3_t grad_v_pix_down = {scalar_t(0), scalar_t(0), scalar_t(0)};

      const scalar3_t center_normal = get_tri_normal(v_pix_n, vp_sV, vp_sC, vi_c0, vi_c1, vi_c2);
      const scalar3_t right_normal = get_tri_normal(v_pix_n, vp_sV, vp_sC, vi_r0, vi_r1, vi_r2);
      const scalar3_t down_normal = get_tri_normal(v_pix_n, vp_sV, vp_sC, vi_d0, vi_d1, vi_d2);

      if (!horiz_int) {
        grad_v_pix_center.x += (!c_valid || r_over_l || horiz_adjacent) ? scalar_t(0) : grad_dot_x;
        grad_v_pix_right.x += (!r_valid || l_over_r || horiz_adjacent) ? scalar_t(0) : grad_dot_x;
      } else {
        scalar2_t dpx_dr = get_dp_dr<scalar_t>(
            {center_normal.x, center_normal.z},
            {right_normal.x, right_normal.z},
            static_cast<scalar_t>(max_dp_dr));
        grad_v_pix_center.x += grad_dot_x * dpx_dr.x;
        grad_v_pix_center.z += grad_dot_x * dpx_dr.y;

        dpx_dr = get_dp_dr<scalar_t>(
            {right_normal.x, right_normal.z},
            {center_normal.x, center_normal.z},
            static_cast<scalar_t>(max_dp_dr));
        grad_v_pix_right.x += grad_dot_x * dpx_dr.x;
        grad_v_pix_right.z += grad_dot_x * dpx_dr.y;
      }

      if (!vert_int) {
        grad_v_pix_center.y += (!c_valid || d_over_u || vert_adjacent) ? scalar_t(0) : grad_dot_y;
        grad_v_pix_down.y += (!d_valid || u_over_d || vert_adjacent) ? scalar_t(0) : grad_dot_y;
      } else {
        scalar2_t dpy_dr = get_dp_dr<scalar_t>(
            {center_normal.y, center_normal.z},
            {down_normal.y, down_normal.z},
            static_cast<scalar_t>(max_dp_dr));
        grad_v_pix_center.y += grad_dot_y * dpy_dr.x;
        grad_v_pix_center.z += grad_dot_y * dpy_dr.y;

        dpy_dr = get_dp_dr<scalar_t>(
            {down_normal.y, down_normal.z},
            {center_normal.y, center_normal.z},
            static_cast<scalar_t>(max_dp_dr));
        grad_v_pix_down.y += grad_dot_y * dpy_dr.x;
        grad_v_pix_down.z += grad_dot_y * dpy_dr.y;
      }

      // Write gradients — atomics needed because neighboring pixels write
      // to overlapping locations (center writes to (x,y), right neighbor
      // also writes to (x,y) as its "center" pixel, etc.)
      scalar_t* gvpi_n = grad_v_pix_img_ptr + gvpi_sN * n;

      // center
      auto* ptr_c = gvpi_n + (y + 0) * gvpi_sH + (x + 0) * gvpi_sW;
      atomic_add(ptr_c + 0 * gvpi_sC, -grad_v_pix_center.x);
      atomic_add(ptr_c + 1 * gvpi_sC, -grad_v_pix_center.y);
      atomic_add(ptr_c + 2 * gvpi_sC, -grad_v_pix_center.z);

      // right
      auto* ptr_r = gvpi_n + (y + 0) * gvpi_sH + (x + 1) * gvpi_sW;
      atomic_add(ptr_r + 0 * gvpi_sC, -grad_v_pix_right.x);
      atomic_add(ptr_r + 1 * gvpi_sC, -grad_v_pix_right.y);
      atomic_add(ptr_r + 2 * gvpi_sC, -grad_v_pix_right.z);

      // down
      auto* ptr_d = gvpi_n + (y + 1) * gvpi_sH + (x + 0) * gvpi_sW;
      atomic_add(ptr_d + 0 * gvpi_sC, -grad_v_pix_down.x);
      atomic_add(ptr_d + 1 * gvpi_sC, -grad_v_pix_down.y);
      atomic_add(ptr_d + 2 * gvpi_sC, -grad_v_pix_down.z);
    }
  });
}

} // namespace

torch::Tensor edge_grad_estimator_cpu_fwd(
    const torch::Tensor& v_pix,
    const torch::Tensor& v_pix_img,
    const torch::Tensor& vi,
    const torch::Tensor& img,
    const torch::Tensor& index_img,
    double /*max_dp_dr*/) {
  TORCH_CHECK(
      v_pix.defined() && v_pix_img.defined() && vi.defined() && img.defined() &&
          index_img.defined(),
      "edge_grad_estimator(): expected all inputs to be defined");
  TORCH_CHECK(
      v_pix.device().is_cpu() && v_pix_img.device().is_cpu() && vi.device().is_cpu() &&
          img.device().is_cpu() && index_img.device().is_cpu(),
      "edge_grad_estimator(): expected all inputs to be on CPU");
  TORCH_CHECK(
      v_pix.is_floating_point() && v_pix_img.is_floating_point() && img.is_floating_point(),
      "edge_grad_estimator(): expected v_pix, v_pix_img, and img to have floating point type");
  TORCH_CHECK(vi.dtype() == torch::kInt32, "edge_grad_estimator(): expected vi to have int32 type");
  TORCH_CHECK(
      index_img.dtype() == torch::kInt32,
      "edge_grad_estimator(): expected index_img to have int32 type");
  TORCH_CHECK(
      (v_pix.dim() == 3) && (v_pix_img.dim() == 4) && (vi.dim() == 3) && (img.dim() == 4) &&
          (index_img.dim() == 3),
      "edge_grad_estimator(): dimension mismatch");
  TORCH_CHECK(
      v_pix.size(0) == v_pix_img.size(0) && v_pix.size(0) == img.size(0) &&
          v_pix.size(0) == index_img.size(0),
      "edge_grad_estimator(): batch size mismatch");
  TORCH_CHECK(
      v_pix.size(2) == 3 && v_pix_img.size(1) == 3 && vi.size(2) == 3,
      "edge_grad_estimator(): expected 3-component vertex/face data");
  TORCH_CHECK(
      v_pix_img.size(3) == img.size(3) && v_pix_img.size(3) == index_img.size(2) &&
          v_pix_img.size(2) == img.size(2) && v_pix_img.size(2) == index_img.size(1),
      "edge_grad_estimator(): spatial dimension mismatch");
  return img;
}

torch::Tensor edge_grad_estimator_cpu_backward(
    const torch::Tensor& v_pix,
    const torch::Tensor& img,
    const torch::Tensor& index_img,
    const torch::Tensor& vi,
    const torch::Tensor& grad_outputs,
    double max_dp_dr) {
  auto v_pix_c = v_pix.contiguous();
  auto img_c = img.contiguous();
  auto index_img_c = index_img.contiguous();
  auto vi_c = vi.contiguous();
  auto grad_out_c = grad_outputs.contiguous();

  const auto N = img_c.size(0);
  const auto C = img_c.size(1);
  const auto H = img_c.size(2);
  const auto W = img_c.size(3);
  const auto count = N * H * W;

  auto grad_v_pix_img = torch::zeros({N, 3, H, W}, v_pix.options());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(v_pix_c.scalar_type(), "edge_grad_cpu_backward", [&] {
      edge_grad_backward_cpu_impl<scalar_t>(
          v_pix_c.data_ptr<scalar_t>(),
          img_c.data_ptr<scalar_t>(),
          index_img_c.data_ptr<int32_t>(),
          vi_c.data_ptr<int32_t>(),
          grad_out_c.data_ptr<scalar_t>(),
          grad_v_pix_img.data_ptr<scalar_t>(),
          N,
          C,
          H,
          W,
          /*vp_sN=*/v_pix_c.size(1) * 3,
          /*vp_sV=*/3,
          /*vp_sC=*/1,
          /*vi_sN=*/vi_c.size(1) * 3,
          /*vi_sV=*/3,
          /*vi_sF=*/1,
          /*idx_sN=*/H * W,
          /*idx_sH=*/W,
          /*idx_sW=*/1,
          /*img_sN=*/C * H * W,
          /*img_sC=*/H * W,
          /*img_sH=*/W,
          /*img_sW=*/1,
          /*go_sN=*/C * H * W,
          /*go_sC=*/H * W,
          /*go_sH=*/W,
          /*go_sW=*/1,
          /*gvpi_sN=*/3 * H * W,
          /*gvpi_sC=*/H * W,
          /*gvpi_sH=*/W,
          /*gvpi_sW=*/1,
          max_dp_dr);
    });
  }
  return grad_v_pix_img;
}

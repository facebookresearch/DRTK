// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <torch/types.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <kernel_utils.h>
#include "edge_grad_kernel.h"

using namespace math;

using at::native::fastAtomicAdd;

template <typename scalar_t>
struct TriInfo {
  typedef typename math::TVec2<scalar_t> scalar2_t;

  const scalar2_t p_0;
  const scalar2_t p_1;
  const scalar2_t v_01;
  const scalar2_t v_02;
  const scalar2_t v_12;
  const scalar_t denominator;
};

template <typename scalar_t>
__device__ bool pix_in_tri(const TriInfo<scalar_t>& tri, const int x, const int y) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;

  if (tri.denominator != 0.f) {
    const scalar2_t p = {(scalar_t)x, (scalar_t)y};

    const scalar2_t vp0p = p - tri.p_0;
    const scalar2_t vp1p = p - tri.p_1;

    scalar3_t bary = scalar3_t({
        vp1p.y * tri.v_12.x - vp1p.x * tri.v_12.y,
        vp0p.x * tri.v_02.y - vp0p.y * tri.v_02.x,
        vp0p.y * tri.v_01.x - vp0p.x * tri.v_01.y,
    });
    bary *= sign(tri.denominator);

    const bool on_edge_or_inside = (bary.x >= 0.f) && (bary.y >= 0.f) && (bary.z >= 0.f);

    bool on_edge_0 = bary.x == 0.f;
    bool on_edge_1 = bary.y == 0.f;
    bool on_edge_2 = bary.z == 0.f;

    const bool is_top_left_0 = (tri.denominator > 0)
        ? (tri.v_12.y < 0.f || tri.v_12.y == 0.0f && tri.v_12.x > 0.f)
        : (tri.v_12.y > 0.f || tri.v_12.y == 0.0f && tri.v_12.x < 0.f);
    const bool is_top_left_1 = (tri.denominator > 0)
        ? (tri.v_02.y > 0.f || tri.v_02.y == 0.0f && tri.v_02.x < 0.f)
        : (tri.v_02.y < 0.f || tri.v_02.y == 0.0f && tri.v_02.x > 0.f);
    const bool is_top_left_2 = (tri.denominator > 0)
        ? (tri.v_01.y < 0.f || tri.v_01.y == 0.0f && tri.v_01.x > 0.f)
        : (tri.v_01.y > 0.f || tri.v_01.y == 0.0f && tri.v_01.x < 0.f);

    const bool is_top_left_or_inside = on_edge_or_inside &&
        !(on_edge_0 && !is_top_left_0 || on_edge_1 && !is_top_left_1 ||
          on_edge_2 && !is_top_left_2);
    return is_top_left_or_inside;
  }
  return false;
}

template <typename scalar_t, typename index_t>
__device__ TriInfo<scalar_t>
get_tri_info(const scalar_t* v_ptr, index_t v_sV, index_t v_sC, int3 vi) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  const scalar2_t p_0 = {v_ptr[v_sV * vi.x + v_sC * 0], v_ptr[v_sV * vi.x + v_sC * 1]};
  const scalar2_t p_1 = {v_ptr[v_sV * vi.y + v_sC * 0], v_ptr[v_sV * vi.y + v_sC * 1]};
  const scalar2_t p_2 = {v_ptr[v_sV * vi.z + v_sC * 0], v_ptr[v_sV * vi.z + v_sC * 1]};

  const scalar2_t v_01 = p_1 - p_0;
  const scalar2_t v_02 = p_2 - p_0;
  const scalar2_t v_12 = p_2 - p_1;

  const scalar_t denominator = v_01.x * v_02.y - v_01.y * v_02.x;

  return {p_0, p_1, v_01, v_02, v_12, denominator};
}

template <typename scalar_t, typename index_t>
__device__ math::TVec3<scalar_t>
get_tri_normal(const scalar_t* v_ptr, index_t v_sV, index_t v_sC, int3 vi) {
  typedef typename math::TVec3<scalar_t> scalar3_t;
  const scalar3_t p_0 = {
      v_ptr[v_sV * vi.x + v_sC * 0], v_ptr[v_sV * vi.x + v_sC * 1], v_ptr[v_sV * vi.x + v_sC * 2]};
  const scalar3_t p_1 = {
      v_ptr[v_sV * vi.y + v_sC * 0], v_ptr[v_sV * vi.y + v_sC * 1], v_ptr[v_sV * vi.y + v_sC * 2]};
  const scalar3_t p_2 = {
      v_ptr[v_sV * vi.z + v_sC * 0], v_ptr[v_sV * vi.z + v_sC * 1], v_ptr[v_sV * vi.z + v_sC * 2]};
  return normalize(cross(p_0 - p_2, p_1 - p_0));
}

template <typename scalar_t>
__device__ math::TVec2<scalar_t> get_dp_dr(
    const math::TVec2<scalar_t>& n_varying_,
    const math::TVec2<scalar_t>& n_fixed_,
    scalar_t max_magnitude = scalar_t(0)) {
  /*
      Computes ∂p/∂r — the derivative of the edge position p with respect
      to fragment position r, projected onto a 2D plane (XZ for vertical
      edges, YZ for horizontal edges). See Eqn. 14 and §S.3 of the paper
      "Rasterized Edge Gradients: Handling Discontinuities Differentiably".

      This relates edge movement to fragment movement for the intersection
      case. The result is a 2D vector whose components are used to scatter
      ∂L/∂p (the edge gradient from Eqn. 11) onto the per-fragment gradient
      image (grad_v_pix_img), which is then reduced to per-vertex gradients
      via the backward pass of the interpolate module.

      Args:
        - n_varying_: face normal of the movable triangle, projected onto the
          plane of consideration (XZ or YZ).
        - n_fixed_: face normal of the fixed triangle, projected onto the same
          plane.
        - max_magnitude: maximum allowed magnitude for ‖∂p/∂r‖. When > 0,
          the result is clamped so that |b₁/d| ≤ max_magnitude.
          When 0, no clamping is applied (for comparison with analytic
          solutions or finite differences).
          Default: 0 (no clamping; the caller passes the value from Python).

      Derivation (from §S.3 of the paper):
        Let n̂_v, n̂_f be the normalized projected normals. Define:
          b = (−n̂_f₂, n̂_f₁)    — n̂_f rotated 90°, in-plane tangent of
                                   the fixed primitive
          d = b · n̂_v           — 2D cross product n̂_f × n̂_v = sin(α)
                                   where α is the angle between the normals
        The edge displacement s lies along b (the edge can only slide along
        the fixed primitive). Projecting onto the x-axis (or y-axis) gives:
          ∂p/∂r = (b₁ / d) · n̂_v

      Numerical stability — clamping ‖∂p/∂r‖ to max_magnitude M:

        The edge gradient estimator uses a first-order linear model:

          L(p + Δp) ≈ L(p) + G · Δp

        where G = ∂L/∂p is the upstream edge gradient (Eqn. 11) and Δp is
        the edge displacement. This linear approximation is only valid for
        |Δp| ≤ 1 pixel — beyond that the edge crosses into a new pixel pair,
        the edge classification changes, and we would need to re-render.

        The full fragment gradient is:

          ∂L/∂r = G · ∂p/∂r

        The magnitude ‖∂p/∂r‖ = |b₁/d| represents the amplification: how
        many pixels of edge movement per pixel of fragment movement. When
        the two triangles are nearly coplanar (d = sin(α) → 0), this blows
        up — a tiny fragment perturbation predicts an edge sweep of millions
        of pixels, far outside the linear model's domain of validity.

        We clamp the amplification to at most M by ensuring |d| ≥ |b₁|/M:

          d_safe = sign(d) · max(|d|, |b₁| / M)

        This bounds ‖∂p/∂r‖ ≤ M: one pixel of fragment movement predicts at
        most M pixels of edge movement. For well-separated triangles where
        |b₁/d| < M already, the clamp has no effect. It only engages for
        nearly-coplanar geometry where the unclamped magnitude exceeds M.

        Note on the angle interpretation: v_pix has x,y in pixel units but
        z in world (camera-space) units. This mismatch squishes the
        projected normals, making d = sin(α) much smaller than the naive
        geometric angle would suggest. The effective world-space dihedral
        angle threshold is approximately θ ≈ f / (M · Z), where f is the
        focal length in pixels and Z is depth. So M = 1e4 (default) with
        f=500, Z=5 clips at θ ≈ 0.6°, but with f=5000, Z=1 clips at
        θ ≈ 29°. Ideally, the focal length would be passed to this
        function to rescale the z-component to pixel units, making the
        threshold independent of camera parameters.

      See: "Rasterized Edge Gradients: Handling Discontinuities
      Differentiably" for the base derivation. The clamping is an extension
      that stabilizes the gradient for nearly-coplanar triangle pairs.
  */
  typedef typename math::TVec2<scalar_t> scalar2_t;

  const auto n_varying = normalize(n_varying_);
  const auto n_fixed = normalize(n_fixed_);
  const scalar2_t b = {-n_fixed.y, n_fixed.x};
  const auto d = dot(b, n_varying);
  if (max_magnitude > scalar_t(0)) {
    // Clamp |b.x/d| ≤ M by ensuring |d| ≥ |b.x|/M.
    const auto abs_d = abs(d);
    const auto abs_bx_over_M = abs(b.x) / max_magnitude;
    // When both d and b.x are zero (n_fixed along first axis), max(0,0)=0.
    // epsclamp prevents 0/0; the result is 0 anyway since b.x=0.
    const auto safe_d =
        (d >= scalar_t(0) ? scalar_t(1) : scalar_t(-1)) * epsclamp(max(abs_d, abs_bx_over_M));
    return (b.x / safe_d) * n_varying;
  } else {
    return b.x / epsclamp(d) * n_varying;
  }
}

template <typename scalar_t, typename index_t>
__device__ math::TVec3<scalar_t> load_vec3_if_valid(
    const scalar_t* __restrict ptr,
    index_t stride,
    bool valid,
    const math::TVec3<scalar_t>& def) {
  if (valid) {
    return {ptr[0 * stride], ptr[1 * stride], ptr[2 * stride]};
  }
  return def;
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void edge_grad_backward_kernel(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> v_pix,
    TensorInfo<scalar_t, index_t> img,
    TensorInfo<int32_t, index_t> index_img,
    TensorInfo<int32_t, index_t> vi,
    TensorInfo<scalar_t, index_t> grad_output,
    TensorInfo<scalar_t, index_t> grad_v_pix_img,
    const index_t memory_span,
    const scalar_t max_dp_dr) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;

  const index_t v_pix_sN = v_pix.strides[0];
  const index_t v_pix_sV = v_pix.strides[1];
  const index_t v_pix_sC = v_pix.strides[2];

  const index_t C = img.sizes[1];
  const index_t H = img.sizes[2];
  const index_t W = img.sizes[3];
  const index_t V = v_pix.sizes[1];

  const index_t index_img_sN = index_img.strides[0];
  const index_t index_img_sH = index_img.strides[1];
  const index_t index_img_sW = index_img.strides[2];

  const index_t img_sN = img.strides[0];
  const index_t img_sC = img.strides[1];
  const index_t img_sH = img.strides[2];
  const index_t img_sW = img.strides[3];

  const index_t grad_output_sN = grad_output.strides[0];
  const index_t grad_output_sC = grad_output.strides[1];
  const index_t grad_output_sH = grad_output.strides[2];
  const index_t grad_output_sW = grad_output.strides[3];

  const index_t grad_v_pix_img_sN = grad_v_pix_img.strides[0];
  const index_t grad_v_pix_img_sC = grad_v_pix_img.strides[1];
  const index_t grad_v_pix_img_sH = grad_v_pix_img.strides[2];
  const index_t grad_v_pix_img_sW = grad_v_pix_img.strides[3];

  const index_t vi_sN = vi.strides[0];
  const index_t vi_sV = vi.strides[1];
  const index_t vi_sF = vi.strides[2];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t x = index % W;
    const index_t y = (index / W) % H;
    const index_t n = index / (H * W);

    if (x < (W - 1) && y < (H - 1)) {
      //   center-right-down (CRD)
      //
      //   *--------*--------*
      //   | center |  right |
      //   | (0, 0) | (1, 0) |
      //   *--------*--------*
      //   | down   |
      //   | (0, 1) |
      //   *--------*

      // Computing indicator variables
      // BEGIN
      // triangle indices of CRD pixels
      const int32_t* __restrict index_img_ptr = index_img.data + n * index_img_sN;
      const int32_t center_index = index_img_ptr[(y + 0) * index_img_sH + (x + 0) * index_img_sW];
      const int32_t right_index = index_img_ptr[(y + 0) * index_img_sH + (x + 1) * index_img_sW];
      const int32_t down_index = index_img_ptr[(y + 1) * index_img_sH + (x + 0) * index_img_sW];

      // valid mask
      const bool c_valid = (center_index >= 0);
      const bool r_valid = (right_index >= 0);
      const bool d_valid = (down_index >= 0);

      // vertex indices of triangles of CRD pixels
      // 0,0,0 - if not valid
      const int3 vi_pt_center = load_vec3_if_valid<int32_t, index_t>(
          vi.data + n * vi_sN + center_index * vi_sV, vi_sF, c_valid, {0, 0, 0});
      const int3 vi_pt_right = load_vec3_if_valid<int32_t, index_t>(
          vi.data + n * vi_sN + right_index * vi_sV, vi_sF, r_valid, {0, 0, 0});
      const int3 vi_pt_down = load_vec3_if_valid<int32_t, index_t>(
          vi.data + n * vi_sN + down_index * vi_sV, vi_sF, d_valid, {0, 0, 0});

      // center <-> right differ
      const bool lr_diff = (center_index != right_index);
      // center <-> down differ
      const bool ud_diff = (center_index != down_index);

      // if horizontal pair (vertical edge) composed of two triangles
      const bool x_both_valid = c_valid && r_valid;
      // if vertical pair (horizontal edge) composed of two triangles
      const bool y_both_valid = c_valid && d_valid;

      // Get CRD triangle info
      const scalar_t* __restrict v_pix_ptr = v_pix.data + n * v_pix_sN;
      const auto tri_center = get_tri_info(v_pix_ptr, v_pix_sV, v_pix_sC, vi_pt_center);
      const auto tri_right = get_tri_info(v_pix_ptr, v_pix_sV, v_pix_sC, vi_pt_right);
      const auto tri_down = get_tri_info(v_pix_ptr, v_pix_sV, v_pix_sC, vi_pt_down);

      // Compute indicators of edge type
      const bool center_pix_in_right_tri = lr_diff && x_both_valid && pix_in_tri(tri_right, x, y);
      const bool right_pix_in_center_tri =
          lr_diff && x_both_valid && pix_in_tri(tri_center, x + 1, y);
      const bool center_pix_in_down_tri = ud_diff && y_both_valid && pix_in_tri(tri_down, x, y);
      const bool down_pix_in_center_tri =
          ud_diff && y_both_valid && pix_in_tri(tri_center, x, y + 1);

      // Overlap flags
      const bool l_over_r = center_pix_in_right_tri && (!right_pix_in_center_tri);
      const bool r_over_l = right_pix_in_center_tri && (!center_pix_in_right_tri);
      const bool u_over_d = center_pix_in_down_tri && (!down_pix_in_center_tri);
      const bool d_over_u = down_pix_in_center_tri && (!center_pix_in_down_tri);

      // Intersection flags
      const bool horiz_int = center_pix_in_right_tri && right_pix_in_center_tri;
      const bool vert_int = center_pix_in_down_tri && down_pix_in_center_tri;

      // Intersection flags
      const bool horiz_adjacent =
          lr_diff && x_both_valid && (!center_pix_in_right_tri && !right_pix_in_center_tri);
      const bool vert_adjacent =
          ud_diff && y_both_valid && (!center_pix_in_down_tri && !down_pix_in_center_tri);

      // END

      // Compute image gradient dot output gradient from backward
      // This is computed regardless of the edge type as long as there is an edge (lr_diff or
      // ud_diff) BEGIN
      const scalar_t* __restrict img_ptr = img.data + img_sN * n;
      const scalar_t* __restrict grad_output_ptr = grad_output.data + grad_output_sN * n;

      scalar_t grad_dot_x = 0.f;
      scalar_t grad_dot_y = 0.f;
      if (lr_diff) {
        const scalar_t* __restrict img_ptr_right = img_ptr + y * img_sH + (x + 1) * img_sW;
        const scalar_t* __restrict img_ptr_center = img_ptr + y * img_sH + (x + 0) * img_sW;
        const scalar_t* __restrict grad_output_ptr_right =
            grad_output_ptr + y * grad_output_sH + (x + 1) * grad_output_sW;
        const scalar_t* __restrict grad_output_ptr_center =
            grad_output_ptr + y * grad_output_sH + (x + 0) * grad_output_sW;
        for (size_t c = 0; c < C; ++c) {
          grad_dot_x += (img_ptr_right[c * img_sC] - img_ptr_center[c * img_sC]) *
              (0.5f *
               (grad_output_ptr_right[c * grad_output_sC] +
                grad_output_ptr_center[c * grad_output_sC]));
        }
      }
      if (ud_diff) {
        const scalar_t* __restrict img_ptr_down = img_ptr + (y + 1) * img_sH + x * img_sW;
        const scalar_t* __restrict img_ptr_center = img_ptr + (y + 0) * img_sH + x * img_sW;
        const scalar_t* __restrict grad_output_ptr_down =
            grad_output_ptr + (y + 1) * grad_output_sH + x * grad_output_sW;
        const scalar_t* __restrict grad_output_ptr_center =
            grad_output_ptr + (y + 0) * grad_output_sH + x * grad_output_sW;
        for (size_t c = 0; c < C; ++c) {
          grad_dot_y += (img_ptr_down[c * img_sC] - img_ptr_center[c * img_sC]) *
              (0.5f *
               (grad_output_ptr_down[c * grad_output_sC] +
                grad_output_ptr_center[c * grad_output_sC]));
        }
      }
      // END

      scalar3_t grad_v_pix_center = {0.f, 0.f, 0.f};
      scalar3_t grad_v_pix_right = {0.f, 0.f, 0.f};
      scalar3_t grad_v_pix_down = {0.f, 0.f, 0.f};

      const scalar3_t center_normal = get_tri_normal(v_pix_ptr, v_pix_sV, v_pix_sC, vi_pt_center);
      const scalar3_t right_normal = get_tri_normal(v_pix_ptr, v_pix_sV, v_pix_sC, vi_pt_right);
      const scalar3_t down_normal = get_tri_normal(v_pix_ptr, v_pix_sV, v_pix_sC, vi_pt_down);

      if (!horiz_int) {
        grad_v_pix_center.x += (!c_valid || r_over_l || horiz_adjacent) ? 0.f : grad_dot_x;
        grad_v_pix_right.x += (!r_valid || l_over_r || horiz_adjacent) ? 0.f : grad_dot_x;
      } else {
        // Center triangle moves, right fixed.
        scalar2_t dpx_dr = get_dp_dr<scalar_t>(
            {center_normal.x, center_normal.z}, {right_normal.x, right_normal.z}, max_dp_dr);
        grad_v_pix_center.x += grad_dot_x * dpx_dr.x;
        grad_v_pix_center.z += grad_dot_x * dpx_dr.y;

        // Center triangle fixed, right moves.
        dpx_dr = get_dp_dr<scalar_t>(
            {right_normal.x, right_normal.z}, {center_normal.x, center_normal.z}, max_dp_dr);
        grad_v_pix_right.x += grad_dot_x * dpx_dr.x;
        grad_v_pix_right.z += grad_dot_x * dpx_dr.y;
      }

      if (!vert_int) {
        grad_v_pix_center.y += (!c_valid || d_over_u || vert_adjacent) ? 0.f : grad_dot_y;
        grad_v_pix_down.y += (!d_valid || u_over_d || vert_adjacent) ? 0.f : grad_dot_y;
      } else {
        // Center triangle moves, lower fixed.
        scalar2_t dpy_dr = get_dp_dr<scalar_t>(
            {center_normal.y, center_normal.z}, {down_normal.y, down_normal.z}, max_dp_dr);
        grad_v_pix_center.y += grad_dot_y * dpy_dr.x;
        grad_v_pix_center.z += grad_dot_y * dpy_dr.y;

        // Center triangle fixed, lower moves.
        dpy_dr = get_dp_dr<scalar_t>(
            {down_normal.y, down_normal.z}, {center_normal.y, center_normal.z}, max_dp_dr);
        grad_v_pix_down.y += grad_dot_y * dpy_dr.x;
        grad_v_pix_down.z += grad_dot_y * dpy_dr.y;
      }

      // Writing grads out
      // BEGIN
      scalar_t* __restrict grad_v_pix_img_ptr = grad_v_pix_img.data + grad_v_pix_img_sN * n;

      // center
      auto* ptr_c = grad_v_pix_img_ptr + (y + 0) * grad_v_pix_img_sH + (x + 0) * grad_v_pix_img_sW;
      atomicAdd(ptr_c + 0 * grad_v_pix_img_sC, -grad_v_pix_center.x);
      atomicAdd(ptr_c + 1 * grad_v_pix_img_sC, -grad_v_pix_center.y);
      atomicAdd(ptr_c + 2 * grad_v_pix_img_sC, -grad_v_pix_center.z);

      // right
      auto* ptr_r = grad_v_pix_img_ptr + (y + 0) * grad_v_pix_img_sH + (x + 1) * grad_v_pix_img_sW;
      atomicAdd(ptr_r + 0 * grad_v_pix_img_sC, -grad_v_pix_right.x);
      atomicAdd(ptr_r + 1 * grad_v_pix_img_sC, -grad_v_pix_right.y);
      atomicAdd(ptr_r + 2 * grad_v_pix_img_sC, -grad_v_pix_right.z);

      // down
      auto* ptr_d = grad_v_pix_img_ptr + (y + 1) * grad_v_pix_img_sH + (x + 0) * grad_v_pix_img_sW;
      atomicAdd(ptr_d + 0 * grad_v_pix_img_sC, -grad_v_pix_down.x);
      atomicAdd(ptr_d + 1 * grad_v_pix_img_sC, -grad_v_pix_down.y);
      atomicAdd(ptr_d + 2 * grad_v_pix_img_sC, -grad_v_pix_down.z);
      // END
    }
  }
}

template <typename scalar_t, typename index_type>
void edge_grad_estimator_cuda_backward_(
    const int64_t count,
    const torch::Tensor& v_pix,
    const torch::Tensor& img,
    const torch::Tensor& index_img,
    const torch::Tensor& vi,
    const torch::Tensor& grad_outputs,
    const torch::Tensor& grad_v_pix_img,
    double max_dp_dr) {
  edge_grad_backward_kernel<scalar_t, index_type>
      <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<index_type>(count),
          getTensorInfo<scalar_t, index_type>(v_pix),
          getTensorInfo<scalar_t, index_type>(img),
          getTensorInfo<int32_t, index_type>(index_img),
          getTensorInfo<int32_t, index_type>(vi),
          getTensorInfo<scalar_t, index_type>(grad_outputs),
          getTensorInfo<scalar_t, index_type>(grad_v_pix_img),
          grad_v_pix_img.numel(),
          static_cast<scalar_t>(max_dp_dr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor edge_grad_estimator_cuda_backward(
    const torch::Tensor& v_pix,
    const torch::Tensor& img,
    const torch::Tensor& index_img,
    const torch::Tensor& vi,
    const torch::Tensor& grad_outputs,
    double max_dp_dr) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(img));

  const auto N = img.sizes()[0];
  const auto C = img.sizes()[1];
  const auto H = img.sizes()[2];
  const auto W = img.sizes()[3];
  const auto V = v_pix.sizes()[1];
  const auto count = N * H * W;

  auto grad_v_pix_img = torch::zeros({N, 3, H, W}, v_pix.options());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES(v_pix.scalar_type(), "edge_grad_estimator_kernel", [&] {
      if (at::native::canUse32BitIndexMath(v_pix) && at::native::canUse32BitIndexMath(img) &&
          at::native::canUse32BitIndexMath(index_img) && at::native::canUse32BitIndexMath(vi) &&
          at::native::canUse32BitIndexMath(grad_outputs) &&
          at::native::canUse32BitIndexMath(grad_v_pix_img)) {
        edge_grad_estimator_cuda_backward_<scalar_t, int>(
            count, v_pix, img, index_img, vi, grad_outputs, grad_v_pix_img, max_dp_dr);
      } else {
        edge_grad_estimator_cuda_backward_<scalar_t, int64_t>(
            count, v_pix, img, index_img, vi, grad_outputs, grad_v_pix_img, max_dp_dr);
      }
    });
  }
  return grad_v_pix_img;
}

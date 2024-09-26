// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <torch/types.h>

#include <grid_utils.h>
#include <kernel_utils.h>
#include <tensor_list.h>

using namespace math;

constexpr int max_mipmap_count = 11;
constexpr int tex_ndim = 4;
constexpr int uv_jacobian_ndim = 5;

template <typename scalar_t, typename index_t>
__device__ void sample_bilinear(
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& input,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    const TensorInfo<scalar_t, index_t>& output,
    scalar_t alpha,
    const GridSamplerPadding padding_mode,
    bool align_corners) {
  index_t out_sN = output.strides[0];
  index_t out_sC = output.strides[1];
  index_t out_sH = output.strides[2];
  index_t out_sW = output.strides[3];

  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];

  scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
  scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

  // get NE, NW, SE, SW pixel values from (x, y)
  index_t ix_nw = static_cast<index_t>(::floor(ix));
  index_t iy_nw = static_cast<index_t>(::floor(iy));
  index_t ix_ne = ix_nw + 1;
  index_t iy_ne = iy_nw;
  index_t ix_sw = ix_nw;
  index_t iy_sw = iy_nw + 1;
  index_t ix_se = ix_nw + 1;
  index_t iy_se = iy_nw + 1;

  // get surfaces to each neighbor:
  scalar_t nw = (ix_se - ix) * (iy_se - iy);
  scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
  scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
  scalar_t se = (ix - ix_nw) * (iy - iy_nw);

  // calculate bilinear weighted pixel value and set output pixel
  auto inp_ptr_NC = input.data + n * inp_sN;
  auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
  for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
    if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
      *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw * alpha;
    }
    if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
      *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne * alpha;
    }
    if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
      *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw * alpha;
    }
    if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
      *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se * alpha;
    }
  }
}

template <typename scalar_t, typename index_t>
__device__ void sample_bicubic(
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& input,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    const TensorInfo<scalar_t, index_t>& output,
    scalar_t alpha,
    const GridSamplerPadding padding_mode,
    bool align_corners) {
  index_t out_sN = output.strides[0];
  index_t out_sC = output.strides[1];
  index_t out_sH = output.strides[2];
  index_t out_sW = output.strides[3];

  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];

  scalar_t ix = grid_sampler_unnormalize(x, inp_W, align_corners);
  scalar_t iy = grid_sampler_unnormalize(y, inp_H, align_corners);

  // get NE, NW, SE, SW pixel values from (x, y)
  scalar_t ix_nw = ::floor(ix);
  scalar_t iy_nw = ::floor(iy);

  const scalar_t tx = ix - ix_nw;
  const scalar_t ty = iy - iy_nw;

  auto inp_ptr_NC = input.data + n * inp_sN;
  auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
  for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
    scalar_t coefficients[4];

#pragma unroll 4
    for (index_t i = 0; i < 4; ++i) {
      coefficients[i] = cubic_interp1d(
          get_value_bounded<scalar_t>(
              inp_ptr_NC,
              ix_nw - 1,
              iy_nw - 1 + i,
              inp_W,
              inp_H,
              inp_sW,
              inp_sH,
              padding_mode,
              align_corners),
          get_value_bounded<scalar_t>(
              inp_ptr_NC,
              ix_nw + 0,
              iy_nw - 1 + i,
              inp_W,
              inp_H,
              inp_sW,
              inp_sH,
              padding_mode,
              align_corners),
          get_value_bounded<scalar_t>(
              inp_ptr_NC,
              ix_nw + 1,
              iy_nw - 1 + i,
              inp_W,
              inp_H,
              inp_sW,
              inp_sH,
              padding_mode,
              align_corners),
          get_value_bounded<scalar_t>(
              inp_ptr_NC,
              ix_nw + 2,
              iy_nw - 1 + i,
              inp_W,
              inp_H,
              inp_sW,
              inp_sH,
              padding_mode,
              align_corners),
          tx);
    }

    *out_ptr_NCHW +=
        cubic_interp1d(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty) *
        alpha;
  }
}

template <typename scalar_t, typename index_t>
__device__ TVec2<scalar_t> sample_bilinear_backward(
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& grad_input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& grad_output,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    scalar_t alpha,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    index_t grad_input_memory_span) {
  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sH = grad_output.strides[2];
  index_t gOut_sW = grad_output.strides[3];
  index_t gInp_sN = grad_input.strides[0];
  index_t gInp_sC = grad_input.strides[1];
  index_t gInp_sH = grad_input.strides[2];
  index_t gInp_sW = grad_input.strides[3];

  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];

  // multipliers for gradients on ix and iy
  TVec2<scalar_t> gi_mult;
  scalar_t ix =
      grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gi_mult.x);
  scalar_t iy =
      grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &gi_mult.y);

  // get NE, NW, SE, SW pixel values from (x, y)
  index_t ix_nw = static_cast<index_t>(::floor(ix));
  index_t iy_nw = static_cast<index_t>(::floor(iy));
  index_t ix_ne = ix_nw + 1;
  index_t iy_ne = iy_nw;
  index_t ix_sw = ix_nw;
  index_t iy_sw = iy_nw + 1;
  index_t ix_se = ix_nw + 1;
  index_t iy_se = iy_nw + 1;

  // get surfaces to each neighbor:
  scalar_t nw = (ix_se - ix) * (iy_se - iy);
  scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
  scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
  scalar_t se = (ix - ix_nw) * (iy - iy_nw);

  TVec2<scalar_t> gi = {scalar_t(0), scalar_t(0)};
  scalar_t* gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
  index_t NC_offset = n * gInp_sN;
  scalar_t* inp_ptr_NC = input.data + n * inp_sN;
  for (index_t c = 0; c < C;
       ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
    scalar_t gOut = *gOut_ptr_NCHW * alpha;

    // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
    safe_add_2d(
        grad_input.data,
        iy_nw,
        ix_nw,
        gInp_sH,
        gInp_sW,
        inp_H,
        inp_W,
        nw * gOut,
        NC_offset,
        grad_input_memory_span);
    safe_add_2d(
        grad_input.data,
        iy_ne,
        ix_ne,
        gInp_sH,
        gInp_sW,
        inp_H,
        inp_W,
        ne * gOut,
        NC_offset,
        grad_input_memory_span);
    safe_add_2d(
        grad_input.data,
        iy_sw,
        ix_sw,
        gInp_sH,
        gInp_sW,
        inp_H,
        inp_W,
        sw * gOut,
        NC_offset,
        grad_input_memory_span);
    safe_add_2d(
        grad_input.data,
        iy_se,
        ix_se,
        gInp_sH,
        gInp_sW,
        inp_H,
        inp_W,
        se * gOut,
        NC_offset,
        grad_input_memory_span);

    // calculate grad_grid
    if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
      scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
      gi.x -= nw_val * (iy_se - iy) * gOut;
      gi.y -= nw_val * (ix_se - ix) * gOut;
    }
    if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
      scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
      gi.x += ne_val * (iy_sw - iy) * gOut;
      gi.y -= ne_val * (ix - ix_sw) * gOut;
    }
    if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
      scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
      gi.x -= sw_val * (iy - iy_ne) * gOut;
      gi.y += sw_val * (ix_ne - ix) * gOut;
    }
    if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
      scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
      gi.x += se_val * (iy - iy_nw) * gOut;
      gi.y += se_val * (ix - ix_nw) * gOut;
    }
  }
  return gi_mult * gi;
}

template <typename scalar_t, typename index_t>
__device__ TVec2<scalar_t> sample_bicubic_backward(
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& grad_input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& grad_output,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    scalar_t alpha,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    index_t grad_input_memory_span) {
  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sH = grad_output.strides[2];
  index_t gOut_sW = grad_output.strides[3];
  index_t gInp_sN = grad_input.strides[0];
  index_t gInp_sC = grad_input.strides[1];
  index_t gInp_sH = grad_input.strides[2];
  index_t gInp_sW = grad_input.strides[3];

  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];

  // multipliers for gradients on ix and iy
  TVec2<scalar_t> gi_mult;

  scalar_t ix = grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &gi_mult.x);
  scalar_t iy = grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &gi_mult.y);

  scalar_t ix_nw = ::floor(ix);
  scalar_t iy_nw = ::floor(iy);

  const scalar_t tx = ix - ix_nw;
  const scalar_t ty = iy - iy_nw;

  scalar_t x_coeffs[4];
  scalar_t y_coeffs[4];
  scalar_t x_coeffs_grad[4];
  scalar_t y_coeffs_grad[4];

  get_cubic_upsampling_coefficients<scalar_t>(x_coeffs, tx);
  get_cubic_upsampling_coefficients<scalar_t>(y_coeffs, ty);
  get_cubic_coefficients_grad<scalar_t>(x_coeffs_grad, tx);
  get_cubic_coefficients_grad<scalar_t>(y_coeffs_grad, ty);

  TVec2<scalar_t> gi = {scalar_t(0), scalar_t(0)};

  scalar_t* gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
  index_t NC_offset = n * gInp_sN;
  scalar_t* inp_ptr_NC = input.data + n * inp_sN;
  for (index_t c = 0; c < C;
       ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
    scalar_t gOut = *gOut_ptr_NCHW * alpha;

#pragma unroll 4
    for (index_t i = 0; i < 4; ++i) {
#pragma unroll 4
      for (index_t j = 0; j < 4; ++j) {
        // set input gradient. See Note [Passing pointer and offset to fastAtomicAdd].
        add_value_bounded<scalar_t>(
            grad_input.data,
            ix_nw - 1 + i,
            iy_nw - 1 + j,
            inp_W,
            inp_H,
            gInp_sW,
            gInp_sH,
            gOut * x_coeffs[i] * y_coeffs[j],
            padding_mode,
            align_corners,
            NC_offset,
            grad_input_memory_span);

        // set grid gradient
        scalar_t val = get_value_bounded<scalar_t>(
            inp_ptr_NC,
            ix_nw - 1 + i,
            iy_nw - 1 + j,
            inp_W,
            inp_H,
            inp_sW,
            inp_sH,
            padding_mode,
            align_corners);

        gi -= gOut * val *
            TVec2<scalar_t>({x_coeffs_grad[i] * y_coeffs[j], y_coeffs_grad[j] * x_coeffs[i]});
      }
    }
  }
  return gi_mult * gi;
}

template <typename scalar_t, typename index_t, GridSamplerInterpolation interpolation_mode>
C10_LAUNCH_BOUNDS_1(256)
__global__ void mipmap_aniso_grid_sampler_2d_kernel(
    const index_t nthreads,
    TensorInfoList<scalar_t, index_t, max_mipmap_count, tex_ndim> inputs,
    const int mipmaps,
    TensorInfo<scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> vt_dxdy_img,
    TensorInfo<scalar_t, index_t> output,
    const GridSamplerPadding padding_mode,
    int max_aniso,
    bool align_corners,
    bool force_max_aniso,
    bool clip_grad) {
  align_corners = false;
  index_t C = output.sizes[1];
  index_t inp_H = inputs[0].sizes[2];
  index_t inp_W = inputs[0].sizes[3];
  index_t out_H = grid.sizes[1];
  index_t out_W = grid.sizes[2];

  index_t grid_sN = grid.strides[0];
  index_t grid_sH = grid.strides[1];
  index_t grid_sW = grid.strides[2];
  index_t grid_sCoor = grid.strides[3];
  index_t vt_dxdy_img_sN = vt_dxdy_img.strides[0];
  index_t vt_dxdy_img_sH = vt_dxdy_img.strides[1];
  index_t vt_dxdy_img_sW = vt_dxdy_img.strides[2];
  index_t vt_dxdy_img_s3 = vt_dxdy_img.strides[3];
  index_t vt_dxdy_img_s4 = vt_dxdy_img.strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t n = index / (out_H * out_W);
    const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
    const index_t vt_dxdy_img_offset = n * vt_dxdy_img_sN + h * vt_dxdy_img_sH + w * vt_dxdy_img_sW;

    // get the corresponding input x, y co-ordinates from grid
    scalar_t u = grid.data[grid_offset];
    scalar_t v = grid.data[grid_offset + grid_sCoor];

    scalar_t dudx = vt_dxdy_img.data[vt_dxdy_img_offset];
    scalar_t dvdx = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s4];

    scalar_t dudy = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s3];
    scalar_t dvdy = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s3 + vt_dxdy_img_s4];

    scalar_t px = pow(pow(abs(dudx * inp_W), 2.0f) + pow(abs(dvdx * inp_H), 2.0f) + 1e-12f, 0.5f);
    scalar_t py = pow(pow(abs(dudy * inp_W), 2.0f) + pow(abs(dvdy * inp_H), 2.0f) + 1e-12f, 0.5f);

    scalar_t p_max = max(px, py);
    scalar_t p_min = min(px, py);

    // # See p.255 of OpenGL Core Profile
    // # N = min(ceil(Pmax/Pmin),maxAniso)
    scalar_t N = min(ceil(p_max / p_min), (scalar_t)max_aniso);
    if (p_min == 0.0 || N == 0) {
      N = 1;
    }

    // Lambda' = log2(Pmax/N)
    scalar_t lambda_ = log2(p_max / N);
    if (isnan(lambda_) || isinf(lambda_)) {
      lambda_ = 0.0f;
    }

    // See eq. 8.15, 8.16
    // Substract small number (1e-6) so that `l` is always < mipmaps - 1
    scalar_t l = min(lambda_, mipmaps - 1 - 1e-6);

    // The following correction is divergence from the specification
    // The reason is that it is typically assumed that the full pyramid is available, but if not,
    // clipping of the level happens as in the line above, which causes taps to be spread with
    // distances higher than the size of the texel. Which in turn causes aliasing and not desirable
    // long-range sampling So if clipping happens, we recompute clipped Pmax and scale gradients
    // accordingly
    if (clip_grad && lambda_ > mipmaps - 1) {
      scalar_t p_max_corrected = exp2(l) * N;
      scalar_t scaling = p_max_corrected / p_max;
      dudx *= scaling;
      dvdx *= scaling;
      dudy *= scaling;
      dvdy *= scaling;
    }

    l = max(l, 0.0);
    auto d1 = (index_t)floor(l);

    scalar_t a = l - (scalar_t)d1;

    index_t N_int = index_t(N);
    if (force_max_aniso) {
      N_int = max_aniso;
    }

    if (px > py) {
      for (int i = 0; i < N_int; ++i) {
        scalar_t u_offset = dudx * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);
        scalar_t v_offset = dvdx * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);

        scalar_t alpha_1 = a / N_int;
        scalar_t alpha_2 = (1.0 - a) / N_int;

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          sample_bilinear(
              inputs[d1],
              u + u_offset,
              v + v_offset,
              w,
              h,
              n,
              C,
              output,
              alpha_2,
              padding_mode,
              align_corners);
          if (mipmaps > 1)
            sample_bilinear(
                inputs[d1 + 1],
                u + u_offset,
                v + v_offset,
                w,
                h,
                n,
                C,
                output,
                alpha_1,
                padding_mode,
                align_corners);
        } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
          sample_bicubic(
              inputs[d1],
              u + u_offset,
              v + v_offset,
              w,
              h,
              n,
              C,
              output,
              alpha_2,
              padding_mode,
              align_corners);
          if (mipmaps > 1)
            sample_bicubic(
                inputs[d1 + 1],
                u + u_offset,
                v + v_offset,
                w,
                h,
                n,
                C,
                output,
                alpha_1,
                padding_mode,
                align_corners);
        }
      }
    } else {
      for (int i = 0; i < N_int; ++i) {
        scalar_t u_offset = dudy * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);
        scalar_t v_offset = dvdy * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);

        scalar_t alpha_1 = a / N_int;
        scalar_t alpha_2 = (1.0 - a) / N_int;

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          sample_bilinear(
              inputs[d1],
              u + u_offset,
              v + v_offset,
              w,
              h,
              n,
              C,
              output,
              alpha_2,
              padding_mode,
              align_corners);
          if (mipmaps > 1)
            sample_bilinear(
                inputs[d1 + 1],
                u + u_offset,
                v + v_offset,
                w,
                h,
                n,
                C,
                output,
                alpha_1,
                padding_mode,
                align_corners);
        } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
          sample_bicubic(
              inputs[d1],
              u + u_offset,
              v + v_offset,
              w,
              h,
              n,
              C,
              output,
              alpha_2,
              padding_mode,
              align_corners);
          if (mipmaps > 1)
            sample_bicubic(
                inputs[d1 + 1],
                u + u_offset,
                v + v_offset,
                w,
                h,
                n,
                C,
                output,
                alpha_1,
                padding_mode,
                align_corners);
        }
      }
    }
  }
}

template <typename scalar_t, typename index_t, GridSamplerInterpolation interpolation_mode>
C10_LAUNCH_BOUNDS_1(256)
__global__ void mipmap_aniso_grid_sampler_2d_backward_kernel(
    const index_t nthreads,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_output,
    TensorInfoList<scalar_t, index_t, max_mipmap_count, tex_ndim> inputs,
    const int mipmaps,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grid,
    TensorInfoCompact<scalar_t, index_t, tex_ndim + 1> vt_dxdy_img,
    TensorInfoList<scalar_t, index_t, max_mipmap_count, tex_ndim>
        grad_inputs, // initialized to zeros
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_grid, // initialized to empty
    const GridSamplerPadding padding_mode,
    int max_aniso,
    bool align_corners,
    bool force_max_aniso,
    bool clip_grad,
    IndexList<index_t, max_mipmap_count> grad_input_memory_span) {
  index_t C = inputs[0].sizes[1];
  index_t inp_H = inputs[0].sizes[2];
  index_t inp_W = inputs[0].sizes[3];
  index_t out_H = grid.sizes[1];
  index_t out_W = grid.sizes[2];
  index_t grid_sN = grid.strides[0];
  index_t grid_sH = grid.strides[1];
  index_t grid_sW = grid.strides[2];
  index_t grid_sCoor = grid.strides[3];

  index_t gGrid_sW = grad_grid.strides[2];

  index_t vt_dxdy_img_sN = vt_dxdy_img.strides[0];
  index_t vt_dxdy_img_sH = vt_dxdy_img.strides[1];
  index_t vt_dxdy_img_sW = vt_dxdy_img.strides[2];
  index_t vt_dxdy_img_s3 = vt_dxdy_img.strides[3];
  index_t vt_dxdy_img_s4 = vt_dxdy_img.strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t n = index / (out_H * out_W);
    const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;
    const index_t vt_dxdy_img_offset = n * vt_dxdy_img_sN + h * vt_dxdy_img_sH + w * vt_dxdy_img_sW;

    // get the corresponding input x, y co-ordinates from grid
    scalar_t u = grid.data[grid_offset];
    scalar_t v = grid.data[grid_offset + grid_sCoor];

    scalar_t dudx = vt_dxdy_img.data[vt_dxdy_img_offset];
    scalar_t dvdx = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s4];

    scalar_t dudy = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s3];
    scalar_t dvdy = vt_dxdy_img.data[vt_dxdy_img_offset + vt_dxdy_img_s3 + vt_dxdy_img_s4];

    scalar_t px = pow(pow(abs(dudx * inp_W), 2.0f) + pow(abs(dvdx * inp_H), 2.0f) + 1e-12f, 0.5f);
    scalar_t py = pow(pow(abs(dudy * inp_W), 2.0f) + pow(abs(dvdy * inp_H), 2.0f) + 1e-12f, 0.5f);

    scalar_t p_max = max(px, py);
    scalar_t p_min = min(px, py);

    // # See p.255 of OpenGL Core Profile
    // # N = min(ceil(Pmax/Pmin),maxAniso)
    scalar_t N = min(ceil(p_max / p_min), (scalar_t)max_aniso);
    if (p_min == 0.0 || N == 0) {
      N = 1;
    }

    // Lambda' = log2(Pmax/N)
    scalar_t lambda_ = log2(p_max / N);
    if (isnan(lambda_) || isinf(lambda_)) {
      lambda_ = 0.0f;
    }

    // See eq. 8.15, 8.16
    // Substract small number (1e-6) so that `l` is always < mipmaps - 1
    scalar_t l = min(lambda_, mipmaps - 1 - 1e-6);

    // The following correction is divergence from the specification
    // The reason is that it is typically assumed that the full pyramid is available, but if not,
    // clipping of the level happens as in the line above, which causes taps to be spread with
    // distances higher than the size of the texel. Which in turn causes aliasing and not desirable
    // long-range sampling So if clipping happens, we recompute clipped Pmax and scale gradients
    // accordingly
    if (clip_grad && lambda_ > mipmaps - 1) {
      scalar_t p_max_corrected = exp2(l) * N;
      scalar_t scaling = p_max_corrected / p_max;
      dudx *= scaling;
      dvdx *= scaling;
      dudy *= scaling;
      dvdy *= scaling;
    }

    l = max(l, 0.0);
    auto d1 = (index_t)floor(l);

    scalar_t a = l - (scalar_t)d1;

    index_t N_int = index_t(N);
    if (force_max_aniso) {
      N_int = max_aniso;
    }

    TVec2<scalar_t> gi_acc = {scalar_t(0), scalar_t(0)};

    if (px > py) {
      for (int i = 0; i < N_int; ++i) {
        scalar_t u_offset = dudx * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);
        scalar_t v_offset = dvdx * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);

        scalar_t alpha_1 = a / N_int;
        scalar_t alpha_2 = (1.0 - a) / N_int;

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          auto ggrad = sample_bilinear_backward(
              inputs[d1],
              grad_inputs[d1],
              grad_output,
              u + u_offset,
              v + v_offset,
              w,
              h,
              n,
              C,
              alpha_2,
              padding_mode,
              align_corners,
              grad_input_memory_span[d1]);
          gi_acc += ggrad;
          if (mipmaps > 1) {
            auto ggrad2 = sample_bilinear_backward(
                inputs[d1 + 1],
                grad_inputs[d1 + 1],
                grad_output,
                u + u_offset,
                v + v_offset,
                w,
                h,
                n,
                C,
                alpha_1,
                padding_mode,
                align_corners,
                grad_input_memory_span[d1 + 1]);
            gi_acc += ggrad2;
          }
        } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
          auto ggrad = sample_bicubic_backward(
              inputs[d1],
              grad_inputs[d1],
              grad_output,
              u + u_offset,
              v + v_offset,
              w,
              h,
              n,
              C,
              alpha_2,
              padding_mode,
              align_corners,
              grad_input_memory_span[d1]);
          gi_acc += ggrad;
          if (mipmaps > 1) {
            auto ggrad2 = sample_bicubic_backward(
                inputs[d1 + 1],
                grad_inputs[d1 + 1],
                grad_output,
                u + u_offset,
                v + v_offset,
                w,
                h,
                n,
                C,
                alpha_1,
                padding_mode,
                align_corners,
                grad_input_memory_span[d1 + 1]);
            gi_acc += ggrad2;
          }
        }
      }
    } else {
      for (int i = 0; i < N_int; ++i) {
        scalar_t u_offset = dudy * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);
        scalar_t v_offset = dvdy * ((i + 1.0) / (N_int + 1.0) * 2.0 - 1.0);

        scalar_t alpha_1 = a / N_int;
        scalar_t alpha_2 = (1.0 - a) / N_int;

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          auto ggrad = sample_bilinear_backward(
              inputs[d1],
              grad_inputs[d1],
              grad_output,
              u + u_offset,
              v + v_offset,
              w,
              h,
              n,
              C,
              alpha_2,
              padding_mode,
              align_corners,
              grad_input_memory_span[d1]);
          gi_acc += ggrad;
          if (mipmaps > 1) {
            auto ggrad2 = sample_bilinear_backward(
                inputs[d1 + 1],
                grad_inputs[d1 + 1],
                grad_output,
                u + u_offset,
                v + v_offset,
                w,
                h,
                n,
                C,
                alpha_1,
                padding_mode,
                align_corners,
                grad_input_memory_span[d1 + 1]);
            gi_acc += ggrad2;
          }
        } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
          auto ggrad = sample_bicubic_backward(
              inputs[d1],
              grad_inputs[d1],
              grad_output,
              u + u_offset,
              v + v_offset,
              w,
              h,
              n,
              C,
              alpha_2,
              padding_mode,
              align_corners,
              grad_input_memory_span[d1]);
          gi_acc += ggrad;
          if (mipmaps > 1) {
            auto ggrad2 = sample_bicubic_backward(
                inputs[d1 + 1],
                grad_inputs[d1 + 1],
                grad_output,
                u + u_offset,
                v + v_offset,
                w,
                h,
                n,
                C,
                alpha_1,
                padding_mode,
                align_corners,
                grad_input_memory_span[d1 + 1]);
            gi_acc += ggrad2;
          }
        }
      }
    }

    // assuming grad_grid is contiguous
    // thus we can
    //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
    //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
    scalar_t* gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
    gGrid_ptr_NHW[0] = gi_acc.x;
    gGrid_ptr_NHW[1] = gi_acc.y;
  }
}

__host__ torch::Tensor mipmap_aniso_grid_sampler_2d_cuda(
    const torch::TensorList& input,
    const torch::Tensor& grid,
    const torch::Tensor& vt_dxdy_img,
    int64_t max_aniso,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool force_max_ansio,
    bool clip_grad) {
  int mipmaps = input.size();
  TORCH_CHECK(
      mipmaps >= 1,
      "mipmap_aniso_grid_sampler_2d(): expected input to have at least one mipmap level");

  TORCH_CHECK(
      input[0].defined() && grid.defined(),
      "mipmap_aniso_grid_sampler_2d(): expected input and grid to not be undefined, but input is ",
      input,
      " and grid is ",
      grid);
  auto input_opt = input[0].options();
  auto grid_opt = grid.options();

  TORCH_CHECK(
      input_opt.device() == grid_opt.device(),
      "mipmap_aniso_grid_sampler_2d(): expected input and grid to be on same device, but input is on ",
      input_opt.device(),
      " and grid is on ",
      grid_opt.device());
  TORCH_CHECK(
      input_opt.dtype() == grid_opt.dtype(),
      "mipmap_aniso_grid_sampler_2d(): expected input and grid to have same dtype, but input has ",
      input_opt.dtype(),
      " and grid has ",
      grid_opt.dtype());
  TORCH_CHECK(
      input_opt.layout() == torch::kStrided && grid_opt.layout() == torch::kStrided,
      "mipmap_aniso_grid_sampler_2d(): expected input and grid to have torch.strided layout, but "
      "input has ",
      input_opt.layout(),
      " and grid has ",
      grid_opt.layout());
  TORCH_CHECK(
      (input[0].dim() == 4) && input[0].dim() == grid.dim() &&
          input[0].dim() + 1 == vt_dxdy_img.dim(),
      "mipmap_aniso_grid_sampler_2d(): expected 4D input and grid with same number of "
      "dimensions and 5D vt_dxdy_img, but got input with sizes ",
      input[0].sizes(),
      " and grid with sizes ",
      grid.sizes(),
      " and vt_dxdy_img with sizes ",
      vt_dxdy_img.sizes());
  TORCH_CHECK(
      input[0].size(0) == grid.size(0) && input[0].size(0) == vt_dxdy_img.size(0),
      "mipmap_aniso_grid_sampler_2d(): expected grid, vt_dxdy_img and input to have same batch size, "
      "but got input with sizes ",
      input[0].sizes(),
      " and grid with sizes ",
      grid.sizes(),
      " and vt_dxdy_img with sizes ",
      vt_dxdy_img.sizes());
  TORCH_CHECK(
      grid.size(-1) == input[0].dim() - 2,
      "mipmap_aniso_grid_sampler_2d(): expected grid to have size ",
      input[0].dim() - 2,
      " in last dimension, but got grid with sizes ",
      grid.sizes());
  TORCH_CHECK(
      vt_dxdy_img.size(-1) == input[0].dim() - 2 && vt_dxdy_img.size(-2) == input[0].dim() - 2,
      "mipmap_aniso_grid_sampler_2d(): expected vt_dxdy_img to have size ",
      input[0].dim() - 2,
      " in last "
      "two dimension, but got grid with sizes ",
      grid.sizes());

  for (int64_t i = 1; i < mipmaps; i++) {
    TORCH_CHECK(
        input_opt.device() == input[i].options().device() &&
            input_opt.dtype() == input[i].options().dtype() &&
            input_opt.layout() == input[i].options().layout() && input[0].dim() == input[i].dim() &&
            input[0].size(0) == input[i].size(0) && input[0].size(1) == input[i].size(1),
        "mipmap_aniso_grid_sampler_2d(): expected all inputs to have same device, dtype, layout, and "
        "first two dimensions");
  }
  for (int64_t i = 2; i < input[0].dim(); i++) {
    TORCH_CHECK(
        input[0].size(i) > 0,
        "grid_sampler(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        input[0].sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input[0]));

  auto N = input[0].size(0);
  auto C = input[0].size(1);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto output = at::zeros({N, C, H, W}, input[0].options());
  int64_t count = N * H * W;

  if (count > 0) {
    // Should be AT_DISPATCH_FLOATING_TYPES_AND_HALF, but half is broken on prod
    AT_DISPATCH_FLOATING_TYPES(input[0].scalar_type(), "mipmap_aniso_grid_sampler_2d_kernel", [&] {
      if (at::native::canUse32BitIndexMath(input[0]) && at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(output)) {
        typedef int index_type;

        TensorInfoList<scalar_t, index_type, max_mipmap_count, tex_ndim> inputs;
        for (int i = 0; i < mipmaps; ++i) {
          inputs[i] = getTensorInfo<scalar_t, index_type>(input[i]);
        }
        if (interpolation_mode == (int64_t)GridSamplerInterpolation::Bilinear) {
          mipmap_aniso_grid_sampler_2d_kernel<
              scalar_t,
              index_type,
              GridSamplerInterpolation::Bilinear>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  static_cast<index_type>(count),
                  inputs,
                  mipmaps,
                  getTensorInfo<scalar_t, index_type>(grid),
                  getTensorInfo<scalar_t, index_type>(vt_dxdy_img),
                  getTensorInfo<scalar_t, index_type>(output),
                  static_cast<GridSamplerPadding>(padding_mode),
                  (int)max_aniso,
                  align_corners,
                  force_max_ansio,
                  clip_grad);
        }
        if (interpolation_mode == (int64_t)GridSamplerInterpolation::Bicubic) {
          mipmap_aniso_grid_sampler_2d_kernel<
              scalar_t,
              index_type,
              GridSamplerInterpolation::Bicubic>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  static_cast<index_type>(count),
                  inputs,
                  mipmaps,
                  getTensorInfo<scalar_t, index_type>(grid),
                  getTensorInfo<scalar_t, index_type>(vt_dxdy_img),
                  getTensorInfo<scalar_t, index_type>(output),
                  static_cast<GridSamplerPadding>(padding_mode),
                  (int)max_aniso,
                  align_corners,
                  force_max_ansio,
                  clip_grad);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;

        TensorInfoList<scalar_t, index_type, max_mipmap_count, tex_ndim> inputs;
        for (int i = 0; i < mipmaps; ++i) {
          inputs[i] = getTensorInfo<scalar_t, index_type>(input[i]);
        }
        if (interpolation_mode == (int64_t)GridSamplerInterpolation::Bilinear) {
          mipmap_aniso_grid_sampler_2d_kernel<
              scalar_t,
              index_type,
              GridSamplerInterpolation::Bilinear>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  static_cast<index_type>(count),
                  inputs,
                  mipmaps,
                  getTensorInfo<scalar_t, index_type>(grid),
                  getTensorInfo<scalar_t, index_type>(vt_dxdy_img),
                  getTensorInfo<scalar_t, index_type>(output),
                  static_cast<GridSamplerPadding>(padding_mode),
                  (int)max_aniso,
                  align_corners,
                  force_max_ansio,
                  clip_grad);
        }
        if (interpolation_mode == (int64_t)GridSamplerInterpolation::Bicubic) {
          mipmap_aniso_grid_sampler_2d_kernel<
              scalar_t,
              index_type,
              GridSamplerInterpolation::Bicubic>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  static_cast<index_type>(count),
                  inputs,
                  mipmaps,
                  getTensorInfo<scalar_t, index_type>(grid),
                  getTensorInfo<scalar_t, index_type>(vt_dxdy_img),
                  getTensorInfo<scalar_t, index_type>(output),
                  static_cast<GridSamplerPadding>(padding_mode),
                  (int)max_aniso,
                  align_corners,
                  force_max_ansio,
                  clip_grad);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return output;
}

__host__ std::tuple<std::vector<torch::Tensor>, torch::Tensor>
mipmap_aniso_grid_sampler_2d_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::TensorList& input,
    const torch::Tensor& grid,
    const torch::Tensor& vt_dxdy_img,
    int64_t max_aniso,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool force_max_ansio,
    bool clip_grad) {
  int mipmaps = input.size();
  auto N = input[0].size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input[0]));

  std::vector<torch::Tensor> grad_input;
  for (int i = 0; i < mipmaps; ++i) {
    grad_input.push_back(at::zeros_like(input[i], LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  }
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  int64_t count = N * H * W;

  if (count > 0) {
    // Should be AT_DISPATCH_FLOATING_TYPES_AND_HALF, but half is broken on prod
    AT_DISPATCH_FLOATING_TYPES(
        input[0].scalar_type(), "mipmap_aniso_grid_sampler_2d_backward_kernel", [&] {
          if (at::native::canUse32BitIndexMath(input[0]) &&
              at::native::canUse32BitIndexMath(grid) &&
              at::native::canUse32BitIndexMath(grad_output)) {
            typedef int index_type;

            TensorInfoList<scalar_t, index_type, max_mipmap_count, tex_ndim> inputs;
            IndexList<index_type, max_mipmap_count> grad_input_memory_span;
            TensorInfoList<scalar_t, index_type, max_mipmap_count, tex_ndim> grad_inputs;
            for (int i = 0; i < mipmaps; ++i) {
              inputs[i] = getTensorInfo<scalar_t, index_type>(input[i]);
              grad_inputs[i] = getTensorInfo<scalar_t, index_type>(grad_input[i]);
              grad_input_memory_span[i] = grad_input[i].numel();
            }

            if (interpolation_mode == (int64_t)GridSamplerInterpolation::Bilinear) {
              mipmap_aniso_grid_sampler_2d_backward_kernel<
                  scalar_t,
                  index_type,
                  GridSamplerInterpolation::Bilinear>
                  <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                      static_cast<index_type>(count),
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_output),
                      inputs,
                      mipmaps,
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grid),
                      getTensorInfoCompact<scalar_t, index_type, uv_jacobian_ndim>(vt_dxdy_img),
                      grad_inputs,
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_grid),
                      static_cast<GridSamplerPadding>(padding_mode),
                      (int)max_aniso,
                      align_corners,
                      force_max_ansio,
                      clip_grad,
                      grad_input_memory_span);
            }
            if (interpolation_mode == (int64_t)GridSamplerInterpolation::Bicubic) {
              mipmap_aniso_grid_sampler_2d_backward_kernel<
                  scalar_t,
                  index_type,
                  GridSamplerInterpolation::Bicubic>
                  <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                      static_cast<index_type>(count),
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_output),
                      inputs,
                      mipmaps,
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grid),
                      getTensorInfoCompact<scalar_t, index_type, uv_jacobian_ndim>(vt_dxdy_img),
                      grad_inputs,
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_grid),
                      static_cast<GridSamplerPadding>(padding_mode),
                      (int)max_aniso,
                      align_corners,
                      force_max_ansio,
                      clip_grad,
                      grad_input_memory_span);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            typedef int64_t index_type;

            TensorInfoList<scalar_t, index_type, max_mipmap_count, tex_ndim> inputs;
            IndexList<index_type, max_mipmap_count> grad_input_memory_span;
            TensorInfoList<scalar_t, index_type, max_mipmap_count, tex_ndim> grad_inputs;
            for (int i = 0; i < mipmaps; ++i) {
              inputs[i] = getTensorInfo<scalar_t, index_type>(input[i]);
              grad_inputs[i] = getTensorInfo<scalar_t, index_type>(grad_input[i]);
              grad_input_memory_span[i] = grad_input[i].numel();
            }

            if (interpolation_mode == (int64_t)GridSamplerInterpolation::Bilinear) {
              mipmap_aniso_grid_sampler_2d_backward_kernel<
                  scalar_t,
                  index_type,
                  GridSamplerInterpolation::Bilinear>
                  <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                      static_cast<index_type>(count),
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_output),
                      inputs,
                      mipmaps,
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grid),
                      getTensorInfoCompact<scalar_t, index_type, uv_jacobian_ndim>(vt_dxdy_img),
                      grad_inputs,
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_grid),
                      static_cast<GridSamplerPadding>(padding_mode),
                      (int)max_aniso,
                      align_corners,
                      force_max_ansio,
                      clip_grad,
                      grad_input_memory_span);
            }
            if (interpolation_mode == (int64_t)GridSamplerInterpolation::Bicubic) {
              mipmap_aniso_grid_sampler_2d_backward_kernel<
                  scalar_t,
                  index_type,
                  GridSamplerInterpolation::Bicubic>
                  <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                      static_cast<index_type>(count),
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_output),
                      inputs,
                      mipmaps,
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grid),
                      getTensorInfoCompact<scalar_t, index_type, uv_jacobian_ndim>(vt_dxdy_img),
                      grad_inputs,
                      getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_grid),
                      static_cast<GridSamplerPadding>(padding_mode),
                      (int)max_aniso,
                      align_corners,
                      force_max_ansio,
                      clip_grad,
                      grad_input_memory_span);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
  }
  return std::make_tuple(grad_input, grad_grid);
}

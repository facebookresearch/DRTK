// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <torch/types.h>

#include <grid_utils.h>
#include <kernel_utils.h>
#include <tensor_list.h>

using namespace math;

constexpr int tex_ndim = 4;

template <typename scalar_t, typename index_t>
__device__ void scatter_bilinear(
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& output,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    index_t output_memory_span) {
  index_t input_sN = input.strides[0];
  index_t input_sC = input.strides[1];
  index_t input_sH = input.strides[2];
  index_t input_sW = input.strides[3];
  index_t output_sN = output.strides[0];
  index_t output_sC = output.strides[1];
  index_t output_sH = output.strides[2];
  index_t output_sW = output.strides[3];

  index_t output_H = output.sizes[2];
  index_t output_W = output.sizes[3];

  scalar_t ix = grid_sampler_compute_source_index(x, output_W, padding_mode, align_corners);
  scalar_t iy = grid_sampler_compute_source_index(y, output_H, padding_mode, align_corners);

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

  const scalar_t* input_ptr_NCHW = input.data + n * input_sN + h * input_sH + w * input_sW;
  index_t NC_offset = n * output_sN;
  for (index_t c = 0; c < C; ++c, NC_offset += output_sC) {
    scalar_t input_value = *(input_ptr_NCHW + c * input_sC);

    // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
    safe_add_2d(
        output.data,
        iy_nw,
        ix_nw,
        output_sH,
        output_sW,
        output_H,
        output_W,
        nw * input_value,
        NC_offset,
        output_memory_span);
    safe_add_2d(
        output.data,
        iy_ne,
        ix_ne,
        output_sH,
        output_sW,
        output_H,
        output_W,
        ne * input_value,
        NC_offset,
        output_memory_span);
    safe_add_2d(
        output.data,
        iy_sw,
        ix_sw,
        output_sH,
        output_sW,
        output_H,
        output_W,
        sw * input_value,
        NC_offset,
        output_memory_span);
    safe_add_2d(
        output.data,
        iy_se,
        ix_se,
        output_sH,
        output_sW,
        output_H,
        output_W,
        se * input_value,
        NC_offset,
        output_memory_span);
  }
}

template <typename scalar_t, typename index_t>
__device__ void scatter_bicubic(
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& output,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    index_t output_memory_span) {
  index_t input_sN = input.strides[0];
  index_t input_sC = input.strides[1];
  index_t input_sH = input.strides[2];
  index_t input_sW = input.strides[3];
  index_t output_sN = output.strides[0];
  index_t output_sC = output.strides[1];
  index_t output_sH = output.strides[2];
  index_t output_sW = output.strides[3];

  index_t output_H = output.sizes[2];
  index_t output_W = output.sizes[3];

  scalar_t ix = grid_sampler_compute_source_index(x, output_W, padding_mode, align_corners);
  scalar_t iy = grid_sampler_compute_source_index(y, output_H, padding_mode, align_corners);

  scalar_t ix_nw = static_cast<index_t>(::floor(ix));
  scalar_t iy_nw = static_cast<index_t>(::floor(iy));

  const scalar_t tx = ix - ix_nw;
  const scalar_t ty = iy - iy_nw;

  scalar_t x_coeffs[4];
  scalar_t y_coeffs[4];

  get_cubic_upsampling_coefficients<scalar_t>(x_coeffs, tx);
  get_cubic_upsampling_coefficients<scalar_t>(y_coeffs, ty);

  const scalar_t* input_ptr_NCHW = input.data + n * input_sN + h * input_sH + w * input_sW;
  index_t NC_offset = n * output_sN;
  for (index_t c = 0; c < C; ++c, NC_offset += output_sC) {
    scalar_t input_value = *(input_ptr_NCHW + c * input_sC);

#pragma unroll 4
    for (index_t i = 0; i < 4; ++i) {
#pragma unroll 4
      for (index_t j = 0; j < 4; ++j) {
        // set input gradient. See Note [Passing pointer and offset to fastAtomicAdd].
        add_value_bounded<scalar_t>(
            output.data,
            ix_nw - 1 + i,
            iy_nw - 1 + j,
            output_W,
            output_H,
            output_sW,
            output_sH,
            input_value * x_coeffs[i] * y_coeffs[j],
            padding_mode,
            align_corners,
            NC_offset,
            output_memory_span);
      }
    }
  }
}

template <typename scalar_t, typename index_t, bool grid_requires_grad, bool input_requires_grad>
__device__ TVec2<scalar_t> scatter_bilinear_backward(
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& grad_input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& grad_output,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    const GridSamplerPadding padding_mode,
    bool align_corners) {
  index_t input_sN = input.strides[0];
  index_t input_sC = input.strides[1];
  index_t input_sH = input.strides[2];
  index_t input_sW = input.strides[3];
  index_t grad_input_sN = grad_input.strides[0];
  index_t grad_input_sC = grad_input.strides[1];
  index_t grad_input_sH = grad_input.strides[2];
  index_t grad_input_sW = grad_input.strides[3];

  index_t out_H = grad_output.sizes[2];
  index_t out_W = grad_output.sizes[3];
  index_t grad_output_sN = grad_output.strides[0];
  index_t grad_output_sC = grad_output.strides[1];
  index_t grad_output_sH = grad_output.strides[2];
  index_t grad_output_sW = grad_output.strides[3];

  // multipliers for gradients on ix and iy
  TVec2<scalar_t> gi_mult;
  scalar_t ix =
      grid_sampler_compute_source_index_set_grad(x, out_W, padding_mode, align_corners, &gi_mult.x);
  scalar_t iy =
      grid_sampler_compute_source_index_set_grad(y, out_H, padding_mode, align_corners, &gi_mult.y);

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

  const scalar_t* input_ptr_NCHW = input.data + n * input_sN + h * input_sH + w * input_sW;

  TVec2<scalar_t> gi = {scalar_t(0), scalar_t(0)};
  auto grad_output_ptr_NC = grad_output.data + n * grad_output_sN;
  auto grad_input_ptr_NCHW =
      grad_input.data + n * grad_input_sN + h * grad_input_sH + w * grad_input_sW;
  for (index_t c = 0; c < C;
       ++c, grad_output_ptr_NC += grad_output_sC, grad_input_ptr_NCHW += grad_input_sC) {
    if (input_requires_grad) {
      auto g_input = scalar_t(0.0);
      if (within_bounds_2d(iy_nw, ix_nw, out_H, out_W)) {
        g_input += grad_output_ptr_NC[iy_nw * grad_output_sH + ix_nw * grad_output_sW] * nw;
      }
      if (within_bounds_2d(iy_ne, ix_ne, out_H, out_W)) {
        g_input += grad_output_ptr_NC[iy_ne * grad_output_sH + ix_ne * grad_output_sW] * ne;
      }
      if (within_bounds_2d(iy_sw, ix_sw, out_H, out_W)) {
        g_input += grad_output_ptr_NC[iy_sw * grad_output_sH + ix_sw * grad_output_sW] * sw;
      }
      if (within_bounds_2d(iy_se, ix_se, out_H, out_W)) {
        g_input += grad_output_ptr_NC[iy_se * grad_output_sH + ix_se * grad_output_sW] * se;
      }
      *grad_input_ptr_NCHW = g_input;
    }
    if (grid_requires_grad) {
      // calculate grad_grid
      scalar_t input_value = *(input_ptr_NCHW + c * input_sC);
      if (within_bounds_2d(iy_nw, ix_nw, out_H, out_W)) {
        scalar_t gOut = grad_output_ptr_NC[iy_nw * grad_output_sH + ix_nw * grad_output_sW];
        gi.x -= input_value * (iy_se - iy) * gOut;
        gi.y -= input_value * (ix_se - ix) * gOut;
      }
      if (within_bounds_2d(iy_ne, ix_ne, out_H, out_W)) {
        scalar_t gOut = grad_output_ptr_NC[iy_ne * grad_output_sH + ix_ne * grad_output_sW];
        gi.x += input_value * (iy_sw - iy) * gOut;
        gi.y -= input_value * (ix - ix_sw) * gOut;
      }
      if (within_bounds_2d(iy_sw, ix_sw, out_H, out_W)) {
        scalar_t gOut = grad_output_ptr_NC[iy_sw * grad_output_sH + ix_sw * grad_output_sW];
        gi.x -= input_value * (iy - iy_ne) * gOut;
        gi.y += input_value * (ix_ne - ix) * gOut;
      }
      if (within_bounds_2d(iy_se, ix_se, out_H, out_W)) {
        scalar_t gOut = grad_output_ptr_NC[iy_se * grad_output_sH + ix_se * grad_output_sW];
        gi.x += input_value * (iy - iy_nw) * gOut;
        gi.y += input_value * (ix - ix_nw) * gOut;
      }
    }
  }
  return gi_mult * gi;
}

template <typename scalar_t, typename index_t, bool grid_requires_grad, bool input_requires_grad>
__device__ TVec2<scalar_t> scatter_bicubic_backward(
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& grad_input,
    const TensorInfoCompact<scalar_t, index_t, tex_ndim>& grad_output,
    scalar_t x,
    scalar_t y,
    const index_t w,
    const index_t h,
    const index_t n,
    const index_t C,
    const GridSamplerPadding padding_mode,
    bool align_corners) {
  index_t input_sN = input.strides[0];
  index_t input_sC = input.strides[1];
  index_t input_sH = input.strides[2];
  index_t input_sW = input.strides[3];
  index_t grad_input_sN = grad_input.strides[0];
  index_t grad_input_sC = grad_input.strides[1];
  index_t grad_input_sH = grad_input.strides[2];
  index_t grad_input_sW = grad_input.strides[3];

  index_t out_H = grad_output.sizes[2];
  index_t out_W = grad_output.sizes[3];
  index_t grad_output_sN = grad_output.strides[0];
  index_t grad_output_sC = grad_output.strides[1];
  index_t grad_output_sH = grad_output.strides[2];
  index_t grad_output_sW = grad_output.strides[3];

  // multipliers for gradients on ix and iy
  TVec2<scalar_t> gi_mult;
  scalar_t ix =
      grid_sampler_compute_source_index_set_grad(x, out_W, padding_mode, align_corners, &gi_mult.x);
  scalar_t iy =
      grid_sampler_compute_source_index_set_grad(y, out_H, padding_mode, align_corners, &gi_mult.y);

  // get NE, NW, SE, SW pixel values from (x, y)
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

  const scalar_t* input_ptr_NCHW = input.data + n * input_sN + h * input_sH + w * input_sW;

  TVec2<scalar_t> gi = {scalar_t(0), scalar_t(0)};
  auto grad_output_ptr_NC = grad_output.data + n * grad_output_sN;
  auto grad_input_ptr_NCHW =
      grad_input.data + n * grad_input_sN + h * grad_input_sH + w * grad_input_sW;

  for (index_t c = 0; c < C;
       ++c, grad_output_ptr_NC += grad_output_sC, grad_input_ptr_NCHW += grad_input_sC) {
    scalar_t coefficients[4];
    scalar_t input_value = *(input_ptr_NCHW + c * input_sC);

#pragma unroll 4
    for (index_t i = 0; i < 4; ++i) {
      if (input_requires_grad) {
        coefficients[i] = cubic_interp1d(
            get_value_bounded<scalar_t>(
                grad_output_ptr_NC,
                ix_nw - 1,
                iy_nw - 1 + i,
                out_W,
                out_H,
                grad_output_sW,
                grad_output_sH,
                padding_mode,
                align_corners),
            get_value_bounded<scalar_t>(
                grad_output_ptr_NC,
                ix_nw + 0,
                iy_nw - 1 + i,
                out_W,
                out_H,
                grad_output_sW,
                grad_output_sH,
                padding_mode,
                align_corners),
            get_value_bounded<scalar_t>(
                grad_output_ptr_NC,
                ix_nw + 1,
                iy_nw - 1 + i,
                out_W,
                out_H,
                grad_output_sW,
                grad_output_sH,
                padding_mode,
                align_corners),
            get_value_bounded<scalar_t>(
                grad_output_ptr_NC,
                ix_nw + 2,
                iy_nw - 1 + i,
                out_W,
                out_H,
                grad_output_sW,
                grad_output_sH,
                padding_mode,
                align_corners),
            tx);
        *grad_input_ptr_NCHW =
            cubic_interp1d(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);
      }
      if (grid_requires_grad) {
#pragma unroll 4
        for (index_t j = 0; j < 4; ++j) {
          // set grid gradient
          scalar_t gOut = get_value_bounded<scalar_t>(
              grad_output_ptr_NC,
              ix_nw - 1 + i,
              iy_nw - 1 + j,
              out_W,
              out_H,
              grad_output_sW,
              grad_output_sH,
              padding_mode,
              align_corners);

          gi -= gOut * input_value *
              TVec2<scalar_t>({x_coeffs_grad[i] * y_coeffs[j], y_coeffs_grad[j] * x_coeffs[i]});
        }
      }
    }
  }
  return gi_mult * gi;
}

template <typename scalar_t, typename index_t, GridSamplerInterpolation interpolation_mode>
C10_LAUNCH_BOUNDS_1(256)
__global__ void grid_scatter_2d_kernel(
    const index_t nthreads,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> input,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grid,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> output,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    index_t output_memory_span) {
  index_t C = output.sizes[1];
  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];

  index_t grid_sN = grid.strides[0];
  index_t grid_sH = grid.strides[1];
  index_t grid_sW = grid.strides[2];
  index_t grid_sCoor = grid.strides[3];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % inp_W;
    const index_t h = (index / inp_W) % inp_H;
    const index_t n = index / (inp_H * inp_W);
    const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y co-ordinates from grid
    scalar_t u = grid.data[grid_offset];
    scalar_t v = grid.data[grid_offset + grid_sCoor];
    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      scatter_bilinear(
          input, output, u, v, w, h, n, C, padding_mode, align_corners, output_memory_span);
    } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
      scatter_bicubic(
          input, output, u, v, w, h, n, C, padding_mode, align_corners, output_memory_span);
    }
  }
}

template <
    typename scalar_t,
    typename index_t,
    GridSamplerInterpolation interpolation_mode,
    bool grid_requires_grad,
    bool input_requires_grad>
C10_LAUNCH_BOUNDS_1(256)
__global__ void grid_scatter_2d_backward_kernel(
    const index_t nthreads,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_output,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> input,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grid,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_input,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_grid, // initialized to empty
    const GridSamplerPadding padding_mode,
    bool align_corners) {
  index_t C = input.sizes[1];
  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];

  index_t grid_sN = grid.strides[0];
  index_t grid_sH = grid.strides[1];
  index_t grid_sW = grid.strides[2];
  index_t grid_sCoor = grid.strides[3];

  index_t gGrid_sW = grad_grid.strides[2];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % inp_W;
    const index_t h = (index / inp_W) % inp_H;
    const index_t n = index / (inp_H * inp_W);
    const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y co-ordinates from grid
    scalar_t u = grid.data[grid_offset];
    scalar_t v = grid.data[grid_offset + grid_sCoor];
    scalar_t* gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      auto ggrad =
          scatter_bilinear_backward<scalar_t, index_t, grid_requires_grad, input_requires_grad>(
              input, grad_input, grad_output, u, v, w, h, n, C, padding_mode, align_corners);
      if (grid_requires_grad) {
        gGrid_ptr_NHW[0] = ggrad.x;
        gGrid_ptr_NHW[1] = ggrad.y;
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
      auto ggrad =
          scatter_bicubic_backward<scalar_t, index_t, grid_requires_grad, input_requires_grad>(
              input, grad_input, grad_output, u, v, w, h, n, C, padding_mode, align_corners);
      if (grid_requires_grad) {
        gGrid_ptr_NHW[0] = ggrad.x;
        gGrid_ptr_NHW[1] = ggrad.y;
      }
    }
  }
}

template <typename scalar_t, typename index_t>
__host__ void grid_scatter_2d_dispatch_interpolation_type(
    const index_t nthreads,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> input,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grid,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> output,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    GridSamplerInterpolation interpolation_mode,
    index_t output_memory_span) {
  if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
    grid_scatter_2d_kernel<scalar_t, index_t, GridSamplerInterpolation::Bilinear>
        <<<GET_BLOCKS(nthreads, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            nthreads, input, grid, output, padding_mode, align_corners, output_memory_span);
  } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
    grid_scatter_2d_kernel<scalar_t, index_t, GridSamplerInterpolation::Bicubic>
        <<<GET_BLOCKS(nthreads, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            nthreads, input, grid, output, padding_mode, align_corners, output_memory_span);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename index_t, bool grid_requires_grad, bool input_requires_grad>
__host__ void grid_scatter_2d_backward_dispatch_interpolation_type(
    const index_t nthreads,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_output,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> input,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grid,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_input,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_grid, // initialized to empty
    const GridSamplerPadding padding_mode,
    bool align_corners,
    GridSamplerInterpolation interpolation_mode) {
  if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
    grid_scatter_2d_backward_kernel<
        scalar_t,
        index_t,
        GridSamplerInterpolation::Bilinear,
        grid_requires_grad,
        input_requires_grad>
        <<<GET_BLOCKS(nthreads, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            nthreads, grad_output, input, grid, grad_input, grad_grid, padding_mode, align_corners);
  } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
    grid_scatter_2d_backward_kernel<
        scalar_t,
        index_t,
        GridSamplerInterpolation::Bicubic,
        grid_requires_grad,
        input_requires_grad>
        <<<GET_BLOCKS(nthreads, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            nthreads, grad_output, input, grid, grad_input, grad_grid, padding_mode, align_corners);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename index_t>
__host__ void grid_scatter_2d_backward_dispatch_requires_grad(
    const index_t nthreads,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_output,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> input,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grid,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_input,
    TensorInfoCompact<scalar_t, index_t, tex_ndim> grad_grid, // initialized to empty
    const GridSamplerPadding padding_mode,
    bool align_corners,
    GridSamplerInterpolation interpolation_mode,
    bool grid_requires_grad,
    bool input_requires_grad) {
  if (grid_requires_grad && input_requires_grad) {
    grid_scatter_2d_backward_dispatch_interpolation_type<scalar_t, index_t, true, true>(
        nthreads,
        grad_output,
        input,
        grid,
        grad_input,
        grad_grid,
        padding_mode,
        align_corners,
        interpolation_mode);
  } else if (!grid_requires_grad && input_requires_grad) {
    grid_scatter_2d_backward_dispatch_interpolation_type<scalar_t, index_t, false, true>(
        nthreads,
        grad_output,
        input,
        grid,
        grad_input,
        grad_grid,
        padding_mode,
        align_corners,
        interpolation_mode);
  } else if (grid_requires_grad && !input_requires_grad) {
    grid_scatter_2d_backward_dispatch_interpolation_type<scalar_t, index_t, true, false>(
        nthreads,
        grad_output,
        input,
        grid,
        grad_input,
        grad_grid,
        padding_mode,
        align_corners,
        interpolation_mode);
  }
}

__host__ torch::Tensor grid_scatter_2d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& grid,
    int64_t output_height,
    int64_t output_width,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners) {
  TORCH_CHECK(
      input.defined() && grid.defined(),
      "grid_scatter(): expected input and grid to not be undefined, but input is ",
      input,
      " and grid is ",
      grid);
  auto input_opt = input.options();
  auto grid_opt = grid.options();

  TORCH_CHECK(
      output_height > 0 && output_width > 0,
      "grid_scatter(): expected output_height and output_width to be greater than 0, but output_height is ",
      output_height,
      " and output_width is ",
      output_width);
  TORCH_CHECK(
      input_opt.device() == grid_opt.device() && grid_opt.device().is_cuda(),
      "grid_scatter(): expected input and grid to be on same CUDA device, but input is on ",
      input_opt.device(),
      " and grid is on ",
      grid_opt.device());
  TORCH_CHECK(
      input.is_floating_point() && grid.is_floating_point(),
      "grid_scatter(): expected input and grid to have floating point dtype, but input has ",
      input_opt.dtype(),
      " and grid has ",
      grid_opt.dtype());
  TORCH_CHECK(
      input_opt.layout() == torch::kStrided && grid_opt.layout() == torch::kStrided,
      "grid_scatter(): expected input and grid to have torch.strided layout, but "
      "input has ",
      input_opt.layout(),
      " and grid has ",
      grid_opt.layout());
  TORCH_CHECK(
      (input.dim() == 4) && input.dim() == grid.dim(),
      "grid_scatter(): expected 4D input and grid with same number of "
      "dimensions, but got input with sizes ",
      input.sizes(),
      " and grid with sizes ",
      grid.sizes());
  TORCH_CHECK(
      input.size(0) == grid.size(0) && input.size(2) == grid.size(1) &&
          input.size(3) == grid.size(2),
      "grid_scatter(): expected grid and input to have same batch size, width and height"
      "but got input with sizes ",
      input.sizes(),
      " and grid with sizes ",
      grid.sizes());
  TORCH_CHECK(
      grid.size(-1) == input.dim() - 2,
      "grid_scatter(): expected grid to have size ",
      input[0].dim() - 2,
      " in last dimension, but got grid with sizes ",
      grid.sizes());

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  auto N = input.size(0);
  auto C = input.size(1);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto output = at::zeros({N, C, output_height, output_width}, input.options());
  int64_t count = N * H * W;

  if (count > 0) {
    // Should be AT_DISPATCH_FLOATING_TYPES_AND_HALF, but half is broken on prod
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_scatter_2d_kernel", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(output)) {
        typedef int index_type;

        grid_scatter_2d_dispatch_interpolation_type<scalar_t, index_type>(
            static_cast<index_type>(count),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(input),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grid),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(output),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<index_type>(output.numel()));
      } else {
        typedef int64_t index_type;

        grid_scatter_2d_dispatch_interpolation_type<scalar_t, index_type>(
            static_cast<index_type>(count),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(input),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grid),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(output),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<index_type>(output.numel()));
      }
    });
  }
  return output;
}

__host__ std::tuple<torch::Tensor, torch::Tensor> grid_scatter_2d_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& grid,
    int64_t padding_mode,
    int64_t interpolation_mode,
    bool align_corners,
    bool grid_requires_grad,
    bool input_requires_grad) {
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  int64_t count = N * H * W;

  if (count > 0) {
    // Should be AT_DISPATCH_FLOATING_TYPES_AND_HALF, but half is broken on prod
    AT_DISPATCH_FLOATING_TYPES(input[0].scalar_type(), "grid_scatter_2d_backward_kernel", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(grad_output)) {
        typedef int index_type;

        grid_scatter_2d_backward_dispatch_requires_grad<scalar_t, index_type>(
            static_cast<index_type>(count),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_output),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(input),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grid),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_input),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_grid),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            grid_requires_grad,
            input_requires_grad);
      } else {
        typedef int64_t index_type;

        grid_scatter_2d_backward_dispatch_requires_grad<scalar_t, index_type>(
            static_cast<index_type>(count),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_output),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(input),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grid),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_input),
            getTensorInfoCompact<scalar_t, index_type, tex_ndim>(grad_grid),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            grid_requires_grad,
            input_requires_grad);
      }
    });
  }
  return std::make_tuple(grad_input, grad_grid);
}

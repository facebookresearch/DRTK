// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>
#include <cassert>

#include <cuda_math_helper.h>
#include <grid_utils.h>
#include <kernel_utils.h>

using namespace math;

template <typename scalar_t, typename index_t>
__device__ inline typename math::TVec4<scalar_t> msi_sample_bilinear_cubic(
    const TensorInfo<scalar_t, index_t>& input,
    math::TVec3<scalar_t> uvw) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;
  typedef typename math::TVec4<scalar_t> scalar4_t;

  index_t inp_N = input.sizes[0];
  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];

  int3 size = {(int)inp_W, (int)inp_H, (int)inp_N};

  scalar3_t i_uvw =
      ((uvw + 1.f) * scalar3_t({(float)size.x, (float)size.y, (float)size.z}) - 1.f) / 2.f;
  i_uvw.x = safe_downgrade_to_int_range(clip_coordinates(i_uvw.x, size.x));
  i_uvw.y = safe_downgrade_to_int_range(clip_coordinates(i_uvw.y, size.y));
  i_uvw.z = safe_downgrade_to_int_range(clip_coordinates(i_uvw.z, size.z));

  // get NE, NW, SE, SW pixel values from (x, y)
  index_t ix_nw = static_cast<index_t>(::floor(i_uvw.x));
  index_t iy_nw = static_cast<index_t>(::floor(i_uvw.y));
  index_t iz_nw = static_cast<index_t>(::floor(i_uvw.z));
  index_t ix_ne = ix_nw + 1;
  index_t iy_ne = iy_nw;
  index_t ix_sw = ix_nw;
  index_t iy_sw = iy_nw + 1;
  index_t ix_se = ix_nw + 1;
  index_t iy_se = iy_nw + 1;

  const scalar_t tz = i_uvw.z - iz_nw;

  // get surfaces to each neighbor:
  scalar_t nw = (ix_se - i_uvw.x) * (iy_se - i_uvw.y);
  scalar_t ne = (i_uvw.x - ix_sw) * (iy_sw - i_uvw.y);
  scalar_t sw = (ix_ne - i_uvw.x) * (i_uvw.y - iy_ne);
  scalar_t se = (i_uvw.x - ix_nw) * (i_uvw.y - iy_nw);

  scalar4_t coefficients[4];
#pragma unroll 4
  for (index_t i = 0; i < 4; ++i) {
    scalar_t z = clip_coordinates(iz_nw - 1 + i, size.z);
    int iz = static_cast<int>(z);

    auto inp_ptr_NC = input.data + iz * inp_sN;
    scalar4_t out = {0, 0, 0, 0};

    if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
      auto ptr = inp_ptr_NC + iy_nw * inp_sH + ix_nw * inp_sW;
      out = out + load4(ptr, inp_sC) * nw;
    }
    if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
      auto ptr = inp_ptr_NC + iy_ne * inp_sH + ix_ne * inp_sW;
      out = out + load4(ptr, inp_sC) * ne;
    }
    if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
      auto ptr = inp_ptr_NC + iy_sw * inp_sH + ix_sw * inp_sW;
      out = out + load4(ptr, inp_sC) * sw;
    }
    if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
      auto ptr = inp_ptr_NC + iy_se * inp_sH + ix_se * inp_sW;
      out = out + load4(ptr, inp_sC) * se;
    }
    coefficients[i] = out;
  }
  return cubic_interp1d<scalar_t, 4>(
      coefficients[0], coefficients[1], coefficients[2], coefficients[3], tz);
}

template <typename scalar_t, typename index_t>
__device__ inline void msi_sample_bilinear_cubic_backward(
    const TensorInfo<scalar_t, index_t>& grad_input,
    math::TVec4<scalar_t> grad_output,
    math::TVec3<scalar_t> uvw,
    index_t grad_input_memory_span) {
  typedef typename math::TVec2<scalar_t> scalar2_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;
  typedef typename math::TVec4<scalar_t> scalar4_t;

  index_t gInp_sN = grad_input.strides[0];
  index_t gInp_sC = grad_input.strides[1];
  index_t gInp_sH = grad_input.strides[2];
  index_t gInp_sW = grad_input.strides[3];

  index_t inp_N = grad_input.sizes[0];
  index_t inp_H = grad_input.sizes[2];
  index_t inp_W = grad_input.sizes[3];

  int3 size = {(int)inp_W, (int)inp_H, (int)inp_N};

  scalar3_t i_uvw =
      ((uvw + 1.f) * scalar3_t({(float)size.x, (float)size.y, (float)size.z}) - 1.f) / 2.f;
  i_uvw.x = safe_downgrade_to_int_range(clip_coordinates(i_uvw.x, size.x));
  i_uvw.y = safe_downgrade_to_int_range(clip_coordinates(i_uvw.y, size.y));
  i_uvw.z = safe_downgrade_to_int_range(clip_coordinates(i_uvw.z, size.z));

  // get NE, NW, SE, SW pixel values from (x, y)
  index_t ix_nw = static_cast<index_t>(::floor(i_uvw.x));
  index_t iy_nw = static_cast<index_t>(::floor(i_uvw.y));
  index_t iz_nw = static_cast<index_t>(::floor(i_uvw.z));
  index_t ix_ne = ix_nw + 1;
  index_t iy_ne = iy_nw;
  index_t ix_sw = ix_nw;
  index_t iy_sw = iy_nw + 1;
  index_t ix_se = ix_nw + 1;
  index_t iy_se = iy_nw + 1;

  const scalar_t tz = i_uvw.z - iz_nw;

  // get surfaces to each neighbor:
  scalar_t nw = (ix_se - i_uvw.x) * (iy_se - i_uvw.y);
  scalar_t ne = (i_uvw.x - ix_sw) * (iy_sw - i_uvw.y);
  scalar_t sw = (ix_ne - i_uvw.x) * (i_uvw.y - iy_ne);
  scalar_t se = (i_uvw.x - ix_nw) * (i_uvw.y - iy_nw);

  scalar_t coeffs[4];

  get_cubic_upsampling_coefficients<scalar_t>(coeffs, tz);

#pragma unroll 4
  for (index_t i = 0; i < 4; ++i) {
    scalar_t z = clip_coordinates(iz_nw - 1 + i, size.z);
    int iz = static_cast<int>(z);

    index_t N_offset = iz * gInp_sN;

    // calculate and set grad_input. See Note [Passing pointer and offset to
    // fastAtomicAdd].
    safe_add_2d4(
        grad_input.data,
        gInp_sC,
        iy_nw,
        ix_nw,
        gInp_sH,
        gInp_sW,
        inp_H,
        inp_W,
        nw * grad_output * coeffs[i],
        N_offset,
        grad_input_memory_span);
    safe_add_2d4(
        grad_input.data,
        gInp_sC,
        iy_ne,
        ix_ne,
        gInp_sH,
        gInp_sW,
        inp_H,
        inp_W,
        ne * grad_output * coeffs[i],
        N_offset,
        grad_input_memory_span);
    safe_add_2d4(
        grad_input.data,
        gInp_sC,
        iy_sw,
        ix_sw,
        gInp_sH,
        gInp_sW,
        inp_H,
        inp_W,
        sw * grad_output * coeffs[i],
        N_offset,
        grad_input_memory_span);
    safe_add_2d4(
        grad_input.data,
        gInp_sC,
        iy_se,
        ix_se,
        gInp_sH,
        gInp_sW,
        inp_H,
        inp_W,
        se * grad_output * coeffs[i],
        N_offset,
        grad_input_memory_span);
  }
}

__device__ __host__ __forceinline__ float2 direction_to_equirectangular(float3 d) {
  const float longitude = atan2f(d.z, d.x);
  const float latitude = atan2f(d.y, math::norm(float2{d.x, d.z}));
  constexpr float inv_pi = M_1_PI;

  return float2({longitude, 2 * latitude}) * inv_pi;
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void msi_forward_kernel(
    const index_t nthreads,
    TensorInfo<float, index_t> ray_o,
    TensorInfo<float, index_t> ray_d,
    TensorInfo<scalar_t, index_t> texture,
    TensorInfo<scalar_t, index_t> rgba_img,
    int sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh) {
  typedef typename math::TVec4<scalar_t> scalar4_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;

  const int n_layers = texture.sizes[0];
  const int n_steps = n_layers * sub_step_count;

  const index_t ray_o_sN = ray_o.strides[0];
  const index_t ray_o_sC = ray_o.strides[1];

  const index_t ray_d_sN = ray_d.strides[0];
  const index_t ray_d_sC = ray_d.strides[1];

  const index_t rgba_img_sN = rgba_img.strides[0];
  const index_t rgba_img_sC = rgba_img.strides[1];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    auto rgba_ptr = rgba_img.data + rgba_img_sN * index;

    const float3 r_o = {
        ray_o.data[ray_o_sN * index + ray_o_sC * 0],
        ray_o.data[ray_o_sN * index + ray_o_sC * 1],
        ray_o.data[ray_o_sN * index + ray_o_sC * 2]};

    const float3 r_d = normalize(float3(
        {ray_d.data[ray_d_sN * index + ray_d_sC * 0],
         ray_d.data[ray_d_sN * index + ray_d_sC * 1],
         ray_d.data[ray_d_sN * index + ray_d_sC * 2]}));

    float tc = dot(-r_o, r_d);
    float h2 = dot(r_o, r_o) - tc * tc;

    float step_size = 1.0f / float(n_steps);

    float3 out_v = {0.f, 0.f, 0.f};
    float log_transmit = 0.f;

    for (int i = 0; i < n_steps; ++i) {
      const float a = (float(n_steps - 1 - i) + 0.5f) / float(n_steps);
      const float inv_r = (1.0 - a) * max_inv_r + a * min_inv_r;

      const float r = 1.0f / inv_r;

      float det = r * r - h2;
      if (det < 0.0f)
        continue;

      float t = tc + sqrt(det);
      float3 pos = t * r_d + r_o;

      const float w = 1.f - a * 2.f;

      const float3 uvw = make_float3(direction_to_equirectangular(pos), w);

      auto sample = msi_sample_bilinear_cubic(texture, uvw);

      scalar3_t rgb = {sample.x, sample.y, sample.z};
      float alpha = sample.w;

      if (alpha > 0.0f) {
        const float pcnt = alpha * step_size;
        const float weight = __expf(log_transmit) * (1.f - __expf(-pcnt));
        log_transmit -= pcnt;

        out_v = out_v + weight * math::max(rgb, {0.f, 0.f, 0.f});

        if (__expf(log_transmit) < stop_thresh) {
          log_transmit = -1e3f;
          break;
        }
      }
    }

    rgba_ptr[0 * rgba_img_sC] = out_v.x;
    rgba_ptr[1 * rgba_img_sC] = out_v.y;
    rgba_ptr[2 * rgba_img_sC] = out_v.z;
    rgba_ptr[3 * rgba_img_sC] = log_transmit;
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void msi_backward_kernel(
    const index_t nthreads,
    TensorInfo<float, index_t> ray_o,
    TensorInfo<float, index_t> ray_d,
    TensorInfo<scalar_t, index_t> texture,
    TensorInfo<scalar_t, index_t> texture_grad,
    index_t texture_grad_memory_span,
    TensorInfo<scalar_t, index_t> rgba_img,
    TensorInfo<scalar_t, index_t> rgba_img_grad,
    int sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh) {
  typedef typename math::TVec4<scalar_t> scalar4_t;
  typedef typename math::TVec3<scalar_t> scalar3_t;

  const int n_layers = texture.sizes[0];
  const int n_steps = n_layers * sub_step_count;

  const index_t ray_o_sN = ray_o.strides[0];
  const index_t ray_o_sC = ray_o.strides[1];

  const index_t ray_d_sN = ray_d.strides[0];
  const index_t ray_d_sC = ray_d.strides[1];

  const index_t rgba_img_sN = rgba_img.strides[0];
  const index_t rgba_img_sC = rgba_img.strides[1];
  const index_t rgba_img_grad_sN = rgba_img_grad.strides[0];
  const index_t rgba_img_grad_sC = rgba_img_grad.strides[1];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    auto rgba_ptr = rgba_img.data + rgba_img_sN * index;
    auto rgba_grad_ptr = rgba_img_grad.data + rgba_img_grad_sN * index;

    scalar3_t out_v_grad = {
        rgba_grad_ptr[0 * rgba_img_grad_sC],
        rgba_grad_ptr[1 * rgba_img_grad_sC],
        rgba_grad_ptr[2 * rgba_img_grad_sC]};
    scalar3_t out_v_acc =
        out_v_grad *
        scalar3_t(
            {rgba_ptr[0 * rgba_img_sC], rgba_ptr[1 * rgba_img_sC], rgba_ptr[2 * rgba_img_sC]});

    const float3 r_o = {
        ray_o.data[ray_o_sN * index + ray_o_sC * 0],
        ray_o.data[ray_o_sN * index + ray_o_sC * 1],
        ray_o.data[ray_o_sN * index + ray_o_sC * 2]};

    const float3 r_d = normalize(float3(
        {ray_d.data[ray_d_sN * index + ray_d_sC * 0],
         ray_d.data[ray_d_sN * index + ray_d_sC * 1],
         ray_d.data[ray_d_sN * index + ray_d_sC * 2]}));

    float tc = dot(-r_o, r_d);
    float h2 = dot(r_o, r_o) - tc * tc;

    float step_size = 1.0f / float(n_steps);

    float log_transmit = 0.f;

    for (int i = 0; i < n_steps; ++i) {
      const float a = (float(n_steps - 1 - i) + 0.5f) / float(n_steps);
      const float inv_r = (1.0 - a) * max_inv_r + a * min_inv_r;

      const float r = 1.0f / inv_r;

      float det = r * r - h2;
      if (det < 0.0f)
        continue;

      float t = tc + sqrt(det);
      float3 pos = t * r_d + r_o;

      const float w = 1.f - a * 2.f;

      const float3 uvw = make_float3(direction_to_equirectangular(pos), w);

      auto sample = msi_sample_bilinear_cubic(texture, uvw);

      scalar3_t rgb = {sample.x, sample.y, sample.z};
      float alpha = sample.w;

      if (alpha > 0.0f) {
        const float pcnt = alpha * step_size;
        const float weight = __expf(log_transmit) * (1.f - __expf(-pcnt));
        log_transmit -= pcnt;

        auto rgb_01 = math::max(rgb, {0.f, 0.f, 0.f});
        scalar3_t color_in_01 = scalar3_t(
            {scalar_t(rgb_01.x == rgb.x),
             scalar_t(rgb_01.y == rgb.y),
             scalar_t(rgb_01.z == rgb.z)});

        scalar3_t color_grad = color_in_01 * weight * out_v_grad;

        out_v_acc -= weight * rgb_01 * out_v_grad;

        float alpha_grad =
            sum(rgb_01 * out_v_grad * __expf(-alpha) * __expf(log_transmit) - out_v_acc);

        scalar4_t rgba_grad = make_float4(color_grad, alpha_grad);

        msi_sample_bilinear_cubic_backward(texture_grad, rgba_grad, uvw, texture_grad_memory_span);

        if (__expf(log_transmit) < stop_thresh) {
          log_transmit = -1e3f;
          break;
        }
      }
    }
  }
}

__host__ torch::Tensor msi_forward_cuda(
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& texture,
    int64_t sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh) {
  TORCH_CHECK(sub_step_count > 0, "msi(): expected step_size > 0, but got ", sub_step_count);
  TORCH_CHECK(
      stop_thresh > 0 && stop_thresh < 1,
      "msi(): expected 0 < stop_thresh < 1, but got ",
      stop_thresh);

  TORCH_CHECK(
      min_inv_r > max_inv_r,
      "msi(): expected min_inv_r to be greater than max_inv_r, but "
      "got min_inv_r:",
      min_inv_r,
      " and max_inv_r: ",
      max_inv_r);

  TORCH_CHECK(
      ray_o.defined() && ray_d.defined() && texture.defined(),
      "msi(): expected all inputs not be undefined, but "
      "ray_o is ",
      ray_o,
      ", ray_d is ",
      ray_d,
      ", texture is ",
      texture);

  auto ray_o_opt = ray_o.options();
  auto ray_d_opt = ray_d.options();
  auto texture_opt = texture.options();

  auto device = ray_o_opt.device();
  auto tex_dtype = texture_opt.dtype();
  auto ray_dtype = ray_o_opt.dtype();

  TORCH_CHECK(
      device.is_cuda(), "msi(): expected inputs to be on CUDA device, but got ray_o on ", device);

  const at::cuda::OptionalCUDAGuard device_guard(device);

  TORCH_CHECK(
      device == ray_o_opt.device() && device == ray_d_opt.device() &&
          device == texture_opt.device(),
      "msi(): expected all inputs to be on same device, but input "
      "ray_o is ",
      ray_o_opt.device(),
      ", ray_d is ",
      ray_d_opt.device(),
      ", texture is ",
      texture_opt.device());

  TORCH_CHECK(
      tex_dtype == torch::kFloat64 || tex_dtype == torch::kFloat32 || tex_dtype == torch::kHalf,
      "msi(): expected texture to be of type Double, Float or "
      "Half, but got type ",
      texture_opt.dtype());

  TORCH_CHECK(
      ray_o_opt.dtype() == torch::kFloat32 && ray_d_opt.dtype() == torch::kFloat32,
      "msi(): expected ray_o and ray_d to be of type Float, but "
      "input ray_o is  ",
      ray_o_opt.dtype(),
      " and ray_d is ",
      ray_d_opt.dtype());

  TORCH_CHECK(
      torch::kStrided == ray_o_opt.layout() && torch::kStrided == ray_d_opt.layout() &&
          torch::kStrided == texture_opt.layout(),
      "msi(): expected all inputs to have torch.strided layout, but "
      "ray_o has ",
      ray_o_opt.layout(),
      ", ray_d has ",
      ray_d_opt.layout(),
      ", texture has ",
      texture_opt.layout());

  TORCH_CHECK(
      ray_o.dim() == 2 && ray_d.dim() == 2 && texture.dim() == 4,
      "msi(): expected ray_o and ray_d to have 2 dimensions, "
      "and texture to have 4 dimension, "
      "but got ray_o with size ",
      ray_o.sizes(),
      ", ray_d with size ",
      ray_d.sizes(),
      ", texture with size ",
      texture.sizes());

  TORCH_CHECK(
      ray_o.size(1) == 3 && ray_d.size(1) == 3 && texture.size(1) == 4,
      "msi(): expected ray_o, ray_d to have size 3 along the dimension 1, "
      " and texture to have size 4 along the dimension 1, "
      "but got ray_o with size ",
      ray_o.sizes(),
      ", ray_d with size ",
      ray_d.sizes(),
      ", texture with size ",
      texture.sizes());

  TORCH_CHECK(
      ray_o.size(0) == ray_d.size(0),
      "msi(): expected ray_o, ray_d to have the same size along "
      "the dimension 0, "
      "but got ray_o with size ",
      ray_o.sizes(),
      ", ray_d with size ",
      ray_d.sizes());

  int N = ray_o.size(0);
  auto rgba_img = torch::empty({N, 4}, texture.options());

  if (N > 0) {
    DISPATCH_FLOAT(texture.scalar_type(), "msi_forward_kernel", [&] {
      if (at::native::canUse32BitIndexMath(ray_o) && at::native::canUse32BitIndexMath(ray_d) &&
          at::native::canUse32BitIndexMath(texture)) {
        typedef int index_type;

        msi_forward_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(N, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(N),
                getTensorInfo<float, index_type>(ray_o),
                getTensorInfo<float, index_type>(ray_d),
                getTensorInfo<scalar_t, index_type>(texture),
                getTensorInfo<scalar_t, index_type>(rgba_img),
                (int)sub_step_count,
                min_inv_r,
                max_inv_r,
                stop_thresh);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;

        msi_forward_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(N, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(N),
                getTensorInfo<float, index_type>(ray_o),
                getTensorInfo<float, index_type>(ray_d),
                getTensorInfo<scalar_t, index_type>(texture),
                getTensorInfo<scalar_t, index_type>(rgba_img),
                (int)sub_step_count,
                min_inv_r,
                max_inv_r,
                stop_thresh);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return rgba_img;
}

torch::Tensor msi_backward_cuda(
    const torch::Tensor& rgba_img,
    const torch::Tensor& rgba_img_grad,
    const torch::Tensor& ray_o,
    const torch::Tensor& ray_d,
    const torch::Tensor& texture,
    int64_t sub_step_count,
    double min_inv_r,
    double max_inv_r,
    double stop_thresh) {
  auto ray_o_opt = ray_o.options();
  auto ray_d_opt = ray_d.options();
  auto texture_opt = texture.options();

  auto device = ray_o_opt.device();
  const at::cuda::OptionalCUDAGuard device_guard(device);

  auto tex_dtype = texture_opt.dtype();
  auto ray_dtype = ray_o_opt.dtype();

  int N = ray_o.size(0);
  auto texture_grad = torch::zeros_like(texture);

  if (N > 0) {
    DISPATCH_FLOAT(texture.scalar_type(), "msi_forward_kernel", [&] {
      if (at::native::canUse32BitIndexMath(ray_o) && at::native::canUse32BitIndexMath(ray_d) &&
          at::native::canUse32BitIndexMath(rgba_img) &&
          at::native::canUse32BitIndexMath(rgba_img_grad) &&
          at::native::canUse32BitIndexMath(texture_grad) &&
          at::native::canUse32BitIndexMath(texture)) {
        typedef int index_type;

        index_type texture_grad_memory_span = texture_grad.numel();
        msi_backward_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(N, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(N),
                getTensorInfo<float, index_type>(ray_o),
                getTensorInfo<float, index_type>(ray_d),
                getTensorInfo<scalar_t, index_type>(texture),
                getTensorInfo<scalar_t, index_type>(texture_grad),
                texture_grad_memory_span,
                getTensorInfo<scalar_t, index_type>(rgba_img),
                getTensorInfo<scalar_t, index_type>(rgba_img_grad),
                (int)sub_step_count,
                min_inv_r,
                max_inv_r,
                stop_thresh);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        typedef int64_t index_type;

        index_type texture_grad_memory_span = texture_grad.numel();
        msi_backward_kernel<scalar_t, index_type>
            <<<GET_BLOCKS(N, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<index_type>(N),
                getTensorInfo<float, index_type>(ray_o),
                getTensorInfo<float, index_type>(ray_d),
                getTensorInfo<scalar_t, index_type>(texture),
                getTensorInfo<scalar_t, index_type>(texture_grad),
                texture_grad_memory_span,
                getTensorInfo<scalar_t, index_type>(rgba_img),
                getTensorInfo<scalar_t, index_type>(rgba_img_grad),
                (int)sub_step_count,
                min_inv_r,
                max_inv_r,
                stop_thresh);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return texture_grad;
}

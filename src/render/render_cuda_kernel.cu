#include "render_cuda_kernel.h"

#include <cmath>
#include <cstdio>
#include "helper_math.h"

constexpr float eps = 1e-8f;

inline __device__ __host__ float epsclamp(float v, float _eps = eps) {
  return (v < 0) ? min(v, -_eps) : max(v, _eps);
}

template <bool with_normals>
__global__ void render_forward_kernel(
    int N,
    int H,
    int W,
    int V,
    float3* v2d,
    float2* vt,
    float3* vn,
    int3* vi,
    int3* vti,
    int32_t* indeximg,
    float* depthimg,
    float3* baryimg,
    float2* uvimg,
    float3* vnimg) {
  const int count = N * H * W;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int n = index / (H * W);

    const int32_t indexpt = indeximg[n * H * W + h * W + w];

    if (indexpt >= 0) {
      const int3 vipt = vi[indexpt];

      const float3 p0 = v2d[n * V + vipt.x];
      const float3 p1 = v2d[n * V + vipt.y];
      const float3 p2 = v2d[n * V + vipt.z];

      const float2 v01 = make_float2(p1 - p0);
      const float2 v02 = make_float2(p2 - p0);
      const float nz = epsclamp(v01.x * v02.y - v01.y * v02.x);

      const float2 vp0p = make_float2(w - p0.x, h - p0.y);

      const float bary1 = (vp0p.x * v02.y - vp0p.y * v02.x) / nz;
      const float bary2 = (vp0p.y * v01.x - vp0p.x * v01.y) / nz;
      const float bary0 = 1.f - bary1 - bary2;

      const float w0 = 1.f / epsclamp(p0.z);
      const float w1 = 1.f / epsclamp(p1.z);
      const float w2 = 1.f / epsclamp(p2.z);
      const float zi = 1.f / epsclamp(w0 * bary0 + w1 * bary1 + w2 * bary2);

      const float bary0_3D = w0 * bary0 * zi;
      const float bary1_3D = w1 * bary1 * zi;
      const float bary2_3D = w2 * bary2 * zi;

      const int3 vtipt = vti[indexpt];
      const float2 vt_pt =
          2.f * (bary0_3D * vt[vtipt.x] + bary1_3D * vt[vtipt.y] + bary2_3D * vt[vtipt.z]) - 1.f;
      depthimg[n * H * W + h * W + w] = zi;
      baryimg[n * H * W + h * W + w] = make_float3(bary0_3D, bary1_3D, bary2_3D);
      uvimg[n * H * W + h * W + w] = vt_pt;

      if (with_normals) {
        vnimg[n * H * W + h * W + w] =
            (bary0_3D * vn[n * V + vipt.x] + bary1_3D * vn[n * V + vipt.y] +
             bary2_3D * vn[n * V + vipt.z]);
      }
    } else {
      depthimg[n * H * W + h * W + w] = 0.f;
      baryimg[n * H * W + h * W + w] = make_float3(0.f);
      // Filling unused pixels of vt_img with continuous span of UV space instead of zeros improves
      // backwards pass performance for subsequent grid_sample operations that use vt_img.
      uvimg[n * H * W + h * W + w] = make_float2(w * 2 + 1, h * 2 + 1) / make_float2(W, H) - 1.0f;

      if (with_normals) {
        vnimg[n * H * W + h * W + w] = make_float3(0.f);
      }
    }
  }
}

template <bool with_normals>
__global__ void render_backward_kernel(
    int N,
    int H,
    int W,
    int V,
    float3* v2d,
    float2* vt,
    float3* vn,
    int3* vi,
    int3* vti,
    int32_t* indeximg,
    float* grad_depthimg,
    float3* grad_baryimg,
    float2* grad_uvimg,
    float3* grad_vnimg,
    float* grad_v2d,
    float* grad_vn) {
  const int count = N * H * W;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int n = index / (H * W);

    int32_t indexpt = indeximg[n * H * W + h * W + w];

    if (indexpt >= 0) {
      const int3 vipt = vi[indexpt];

      //
      // Repeat forward pass to compute some checks that weren't needed at that time.
      //
      const float3 p0 = v2d[n * V + vipt.x];
      const float3 p1 = v2d[n * V + vipt.y];
      const float3 p2 = v2d[n * V + vipt.z];

      const float2 v01 = make_float2(p1 - p0);
      const float2 v02 = make_float2(p2 - p0);
      float nz = v01.x * v02.y - v01.y * v02.x;
      const bool nz_clamped = fabs(nz) < eps;
      nz = epsclamp(nz);

      const float2 vp0p = make_float2(w - p0.x, h - p0.y);

      const float bary1_pre = (vp0p.x * v02.y - vp0p.y * v02.x);
      const float bary2_pre = (vp0p.y * v01.x - vp0p.x * v01.y);
      const float bary1 = bary1_pre / nz;
      const float bary2 = bary2_pre / nz;
      const float bary0 = 1.f - bary1 - bary2;

      const bool z0_clamped = fabs(p0.z) < eps;
      const bool z1_clamped = fabs(p1.z) < eps;
      const bool z2_clamped = fabs(p2.z) < eps;
      const float z0eps = epsclamp(p0.z);
      const float z1eps = epsclamp(p1.z);
      const float z2eps = epsclamp(p2.z);

      const float w0 = 1.f / z0eps;
      const float w1 = 1.f / z1eps;
      const float w2 = 1.f / z2eps;
      float bary_wsum = w0 * bary0 + w1 * bary1 + w2 * bary2;
      const bool bary_wsum_clamped = fabs(bary_wsum) < eps;
      bary_wsum = epsclamp(bary_wsum);
      const float zi = 1.f / bary_wsum;

      const float bary0_3D = w0 * bary0 * zi;
      const float bary1_3D = w1 * bary1 * zi;
      const float bary2_3D = w2 * bary2 * zi;

      const int3 vtipt = vti[indexpt];
      const float2 vt_p0 = vt[vtipt.x];
      const float2 vt_p1 = vt[vtipt.y];
      const float2 vt_p2 = vt[vtipt.z];
      const float2 vt_pt = 2.f * (vt_p0 * bary0_3D + vt_p1 * bary1_3D + vt_p2 * bary2_3D) - 1.f;

      //
      // Do backprop, properly handling clamped values.
      //
      const float dL_depthpt = grad_depthimg[n * H * W + h * W + w];
      const float3 dL_barypt = grad_baryimg[n * H * W + h * W + w];
      const float2 dL_vt_pt = grad_uvimg[n * H * W + h * W + w];

      float dL_bary0_3D = 2.f * dot(dL_vt_pt, vt_p0) + dL_barypt.x;
      float dL_bary1_3D = 2.f * dot(dL_vt_pt, vt_p1) + dL_barypt.y;
      float dL_bary2_3D = 2.f * dot(dL_vt_pt, vt_p2) + dL_barypt.z;

      if (with_normals) {
        const float3 vn_0 = vn[n * V + vipt.x];
        const float3 vn_1 = vn[n * V + vipt.y];
        const float3 vn_2 = vn[n * V + vipt.z];
        const float3 vn_pt = (bary0_3D * vn_0 + bary1_3D * vn_1 + bary2_3D * vn_2);
        const float3 dL_vn_pt = grad_vnimg[n * H * W + h * W + w];
        const float3 dL_vn0 = dL_vn_pt * bary0_3D;
        const float3 dL_vn1 = dL_vn_pt * bary1_3D;
        const float3 dL_vn2 = dL_vn_pt * bary2_3D;

        dL_bary0_3D += dot(dL_vn_pt, vn_0);
        dL_bary1_3D += dot(dL_vn_pt, vn_1);
        dL_bary2_3D += dot(dL_vn_pt, vn_2);

        atomicAdd(grad_vn + n * V * 3 + vipt.x * 3 + 0, dL_vn0.x);
        atomicAdd(grad_vn + n * V * 3 + vipt.x * 3 + 1, dL_vn0.y);
        atomicAdd(grad_vn + n * V * 3 + vipt.x * 3 + 2, dL_vn0.z);
        atomicAdd(grad_vn + n * V * 3 + vipt.y * 3 + 0, dL_vn1.x);
        atomicAdd(grad_vn + n * V * 3 + vipt.y * 3 + 1, dL_vn1.y);
        atomicAdd(grad_vn + n * V * 3 + vipt.y * 3 + 2, dL_vn1.z);
        atomicAdd(grad_vn + n * V * 3 + vipt.z * 3 + 0, dL_vn2.x);
        atomicAdd(grad_vn + n * V * 3 + vipt.z * 3 + 1, dL_vn2.y);
        atomicAdd(grad_vn + n * V * 3 + vipt.z * 3 + 2, dL_vn2.z);
      }

      const float dL_zi = dL_bary0_3D * w0 * bary0 + dL_bary1_3D * w1 * bary1 +
          dL_bary2_3D * w2 * bary2 + dL_depthpt;
      const float dL_bary_wsum = bary_wsum_clamped ? 0.f : (-dL_zi / (bary_wsum * bary_wsum));
      const float dL_w0 = dL_bary0_3D * bary0 * zi + dL_bary_wsum * bary0;
      const float dL_w1 = dL_bary1_3D * bary1 * zi + dL_bary_wsum * bary1;
      const float dL_w2 = dL_bary2_3D * bary2 * zi + dL_bary_wsum * bary2;

      const float dL_bary0 = dL_bary0_3D * w0 * zi + dL_bary_wsum * w0;
      const float dL_bary1 = dL_bary1_3D * w1 * zi + dL_bary_wsum * w1 - dL_bary0;
      const float dL_bary2 = dL_bary2_3D * w2 * zi + dL_bary_wsum * w2 - dL_bary0;

      const float2 dL_vp0p = make_float2(
          (dL_bary1 * v02.y - dL_bary2 * v01.y) / nz, (-dL_bary1 * v02.x + dL_bary2 * v01.x) / nz);
      const float _dL_nz = -(dL_bary1 * bary1_pre + dL_bary2 * bary2_pre) / (nz * nz);
      const float dL_nz = nz_clamped ? 0.f : _dL_nz;

      const float2 dL_v02 = make_float2(
          -(dL_bary1 * vp0p.y) / nz - dL_nz * v01.y, (dL_bary1 * vp0p.x) / nz + dL_nz * v01.x);
      const float2 dL_v01 = make_float2(
          (dL_bary2 * vp0p.y) / nz + dL_nz * v02.y, -(dL_bary2 * vp0p.x) / nz - dL_nz * v02.x);

      const float2 dL_p0 = -dL_v02 - dL_v01 - dL_vp0p;
      const float2 dL_p1 = dL_v01;
      const float2 dL_p2 = dL_v02;

      const float dL_z0eps = -dL_w0 / (z0eps * z0eps);
      const float dL_z1eps = -dL_w1 / (z1eps * z1eps);
      const float dL_z2eps = -dL_w2 / (z2eps * z2eps);

      atomicAdd(grad_v2d + n * V * 3 + vipt.x * 3 + 0, dL_p0.x);
      atomicAdd(grad_v2d + n * V * 3 + vipt.x * 3 + 1, dL_p0.y);
      atomicAdd(grad_v2d + n * V * 3 + vipt.x * 3 + 2, z0_clamped ? 0.f : dL_z0eps);
      atomicAdd(grad_v2d + n * V * 3 + vipt.y * 3 + 0, dL_p1.x);
      atomicAdd(grad_v2d + n * V * 3 + vipt.y * 3 + 1, dL_p1.y);
      atomicAdd(grad_v2d + n * V * 3 + vipt.y * 3 + 2, z1_clamped ? 0.f : dL_z1eps);
      atomicAdd(grad_v2d + n * V * 3 + vipt.z * 3 + 0, dL_p2.x);
      atomicAdd(grad_v2d + n * V * 3 + vipt.z * 3 + 1, dL_p2.y);
      atomicAdd(grad_v2d + n * V * 3 + vipt.z * 3 + 2, z2_clamped ? 0.f : dL_z2eps);
    }
  }
}

void render_forward_cuda(
    int N,
    int H,
    int W,
    int V,
    float* v2d,
    float* vt,
    float* vn,
    int32_t* vi,
    int32_t* vti,
    int32_t* indeximg,
    float* depthimg,
    float* baryimg,
    float* uvimg,
    float* vnimg,
    cudaStream_t stream) {
  int count = N * H * W;
  int nthreads = 512;
  int nblocks = (count + nthreads - 1) / nthreads;

  if (vn) {
    render_forward_kernel<true><<<nblocks, nthreads, 0, stream>>>(
        N,
        H,
        W,
        V,
        reinterpret_cast<float3*>(v2d),
        reinterpret_cast<float2*>(vt),
        reinterpret_cast<float3*>(vn),
        reinterpret_cast<int3*>(vi),
        reinterpret_cast<int3*>(vti),
        indeximg,
        depthimg,
        reinterpret_cast<float3*>(baryimg),
        reinterpret_cast<float2*>(uvimg),
        reinterpret_cast<float3*>(vnimg));
  } else {
    render_forward_kernel<false><<<nblocks, nthreads, 0, stream>>>(
        N,
        H,
        W,
        V,
        reinterpret_cast<float3*>(v2d),
        reinterpret_cast<float2*>(vt),
        reinterpret_cast<float3*>(vn),
        reinterpret_cast<int3*>(vi),
        reinterpret_cast<int3*>(vti),
        indeximg,
        depthimg,
        reinterpret_cast<float3*>(baryimg),
        reinterpret_cast<float2*>(uvimg),
        reinterpret_cast<float3*>(vnimg));
  }
}

void render_backward_cuda(
    int N,
    int H,
    int W,
    int V,
    float* v2d,
    float* vt,
    float* vn,
    int32_t* vi,
    int32_t* vti,
    int32_t* indeximg,
    float* grad_depthimg,
    float* grad_baryimg,
    float* grad_uvimg,
    float* grad_vnimg,
    float* grad_v2d,
    float* grad_vn,
    cudaStream_t stream) {
  int count = N * H * W;
  int nthreads = 256;
  int nblocks = (count + nthreads - 1) / nthreads;
  if (vn) {
    render_backward_kernel<true><<<nblocks, nthreads, 0, stream>>>(
        N,
        H,
        W,
        V,
        reinterpret_cast<float3*>(v2d),
        reinterpret_cast<float2*>(vt),
        reinterpret_cast<float3*>(vn),
        reinterpret_cast<int3*>(vi),
        reinterpret_cast<int3*>(vti),
        indeximg,
        grad_depthimg,
        reinterpret_cast<float3*>(grad_baryimg),
        reinterpret_cast<float2*>(grad_uvimg),
        reinterpret_cast<float3*>(grad_vnimg),
        grad_v2d,
        grad_vn);
  } else {
    render_backward_kernel<false><<<nblocks, nthreads, 0, stream>>>(
        N,
        H,
        W,
        V,
        reinterpret_cast<float3*>(v2d),
        reinterpret_cast<float2*>(vt),
        reinterpret_cast<float3*>(vn),
        reinterpret_cast<int3*>(vi),
        reinterpret_cast<int3*>(vti),
        indeximg,
        grad_depthimg,
        reinterpret_cast<float3*>(grad_baryimg),
        reinterpret_cast<float2*>(grad_uvimg),
        reinterpret_cast<float3*>(grad_vnimg),
        grad_v2d,
        grad_vn);
  }
}

#include <torch/types.h>
#include <limits>
#include "../include/kernel_utils.h"
#include "../render/helper_math.h"
#include "rasterizer_kernel.h"

#include <stdio.h>
#include <cmath>
#include <cstdint>

template <typename index_t>
__global__ void packed_rasterize_kernel(
    const index_t nthreads,
    int n,
    int w,
    int h,
    int nprims,
    int nverts,
    const float3* __restrict verts0,
    const int3* __restrict vind,
    unsigned long long int* __restrict packed_index_depth_img0) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t id = index % nprims;
    const index_t ib = (index / nprims);

    const float3* verts = verts0 + ib * nverts;
    unsigned long long int* packed_index_depth_img = packed_index_depth_img0 + ib * h * w;

    const float3 p0 = verts[vind[id].x];
    const float3 p1 = verts[vind[id].y];
    const float3 p2 = verts[vind[id].z];

    // Check Z-coordinate against image plane.
    const float zmin = 1e-8f;
    if (p0.z > zmin && p1.z > zmin && p2.z > zmin) {
      const float2 v01 = make_float2(p1 - p0);
      const float2 v02 = make_float2(p2 - p0);
      const float nz = v01.x * v02.y - v01.y * v02.x;

      // If we wanted backface culling, here is where we would do it. Check n.z < 0.
      if (nz != 0.f) {
        // Compute triangle bounds with extra border.
        int minx = max(0, min(min(int(p0.x), int(p1.x)), int(p2.x)) - 1);
        int miny = max(0, min(min(int(p0.y), int(p1.y)), int(p2.y)) - 1);
        int maxx = min(w - 1, max(max(int(p0.x + .5f), int(p1.x + .5f)), int(p2.x + .5f)) + 1);
        int maxy = min(h - 1, max(max(int(p0.y + .5f), int(p1.y + .5f)), int(p2.y + .5f)) + 1);

        // No need to guard this division since we have already checked the Z
        // coordinates above.
        const float w0 = 1.f / p0.z;
        const float w1 = 1.f / p1.z;
        const float w2 = 1.f / p2.z;

        // Loop over pixels inside triangle bbox.
        for (int y = miny; y <= maxy; ++y) {
          for (int x = minx; x <= maxx; ++x) {
            const int pixel = y * w + x;
            const float2 vp0p = make_float2(x - p0.x, y - p0.y);
            const float bary_1 = (vp0p.x * v02.y - vp0p.y * v02.x) / nz;
            const float bary_2 = (vp0p.y * v01.x - vp0p.x * v01.y) / nz;
            const float bary_0 = 1.f - bary_1 - bary_2;

            if ((bary_1 + bary_2 <= 1.f) && (bary_1 >= 0.f) && (bary_2 >= 0.f)) {
              // No need to guard this division since we already checked that
              // all corners of the triangle have Z > 0, so any point inside
              // the triangle will also have Z > 0.
              const float z = 1.f / (w0 * bary_0 + w1 * bary_1 + w2 * bary_2);

              unsigned long long int packed_val =
                  (static_cast<unsigned long long int>(__float_as_uint(z)) << 32u) |
                  static_cast<unsigned long long int>(id);
              atomicMin(packed_index_depth_img + pixel, packed_val);
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
    const unsigned long long int* __restrict packed_index_depth_img,
    float* __restrict depth_img,
    unsigned int* __restrict index_img) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const unsigned long long int pv = packed_index_depth_img[index];
    depth_img[index] = __uint_as_float(static_cast<unsigned int>(pv >> 32));
    index_img[index] = static_cast<unsigned int>(pv & 0xFFFFFFFF);
  }
}

extern "C" void packed_rasterize_cuda(
    int n,
    int w,
    int h,
    int nprims,
    int nverts,
    const float3* verts,
    const int3* vind,
    float* depth_img,
    int* index_img,
    int* packedindex_img,
    cudaStream_t stream) {
  unsigned long long int* packed_index_depth_img =
      reinterpret_cast<unsigned long long int*>(packedindex_img);
  cudaMemsetAsync(
      packed_index_depth_img, (int64_t)-1, n * w * h * sizeof(unsigned long long int), stream);

  const auto count_rasterize = int64_t(nprims) * n;
  const auto count_unpack = int64_t(w) * h * n;

  // rasterize
  if (count_rasterize) {
    if (count_rasterize < std::numeric_limits<int32_t>::max() &&
        count_unpack < std::numeric_limits<int32_t>::max()) {
      packed_rasterize_kernel<int><<<GET_BLOCKS(count_rasterize, 256), 256, 0, stream>>>(
          count_rasterize, n, w, h, nprims, nverts, verts, vind, packed_index_depth_img);
    } else {
      packed_rasterize_kernel<int64_t><<<GET_BLOCKS(count_rasterize, 256), 256, 0, stream>>>(
          count_rasterize, n, w, h, nprims, nverts, verts, vind, packed_index_depth_img);
    }
  }

  // unpack
  if (count_unpack) {
    if (count_unpack < std::numeric_limits<int32_t>::max()) {
      unpack_kernel<int><<<GET_BLOCKS(count_unpack, 256), 256, 0, stream>>>(
          count_unpack,
          packed_index_depth_img,
          depth_img,
          reinterpret_cast<unsigned int*>(index_img));
    } else {
      unpack_kernel<int64_t><<<GET_BLOCKS(count_unpack, 256), 256, 0, stream>>>(
          count_unpack,
          packed_index_depth_img,
          depth_img,
          reinterpret_cast<unsigned int*>(index_img));
    }
  }
}

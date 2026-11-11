// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the MIT license in the root LICENSE file.

// Included by transparent_render_kernel.cu to share coverage, binning,
// perspective backward, triangle order, and boundary pixel-pair geometry.
#include <ATen/cuda/Atomic.cuh>
#include "source_features_kernel.h"

namespace {

struct SourceParams {
  GeometryParams geometry;
  const int32_t* queries;
  const float* bary;
  const int32_t* adjacency;
  const int32_t* edge_ranges;
  const int32_t* edge_ids;
  const int32_t* offsets;
  const int32_t* faces;
  int32_t query_count;
  int32_t height;
  int32_t width;
};

__device__ __forceinline__ void
source_triangle(const GeometryParams& g, int32_t face, TriangleData& tri) {
  const int32_t n = face / g.face_count;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int32_t id = g.faces[static_cast<int64_t>(face) * 3 + c];
    const int64_t index = static_cast<int64_t>(n) * g.vertex_count + id;
    tri.vertex_id[c] = id;
    tri.x[c] = g.vertices[index * 3];
    tri.y[c] = g.vertices[index * 3 + 1];
    tri.z[c] = g.vertices[index * 3 + 2];
    tri.opacity[c] = g.opacity[index];
  }
}

__device__ __forceinline__ float source_area(const TriangleData& t) {
  return cross2(t.x[1] - t.x[0], t.y[1] - t.y[0], t.x[2] - t.x[0], t.y[2] - t.y[0]);
}

__device__ __forceinline__ void
source_screen_bary(const TriangleData& t, float x, float y, float* b) {
  const float inv = 1.0f / source_area(t);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int a = (c + 1) % 3, d = (c + 2) % 3;
    b[c] = canonical_edge_function(
               t.vertex_id[a], t.vertex_id[d], t.x[a], t.y[a], t.x[d], t.y[d], x, y) *
        inv;
  }
}

__device__ __forceinline__ float source_alpha(
    const TriangleData& t,
    const float* screen,
    float* bary,
    float& inverse_depth,
    float& unclamped) {
  inverse_depth = 0.0f;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    bary[c] = screen[c] / t.z[c];
    inverse_depth += bary[c];
  }
  unclamped = 0.0f;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    bary[c] /= inverse_depth;
  }
  unclamped = t.opacity[0] + bary[1] * (t.opacity[1] - t.opacity[0]) +
      bary[2] * (t.opacity[2] - t.opacity[0]);
  return fminf(fmaxf(unclamped, 0.0f), 1.0f);
}

__global__ void source_edge_keys(GeometryParams g, int32_t count, uint64_t* keys, int32_t* ids) {
  const int32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= count) {
    return;
  }
  keys[edge] = UINT64_MAX;
  ids[edge] = edge;
  const int32_t f = edge / 3, n = f / g.face_count;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int32_t id = g.faces[static_cast<int64_t>(f) * 3 + c];
    if (id < 0 || id >= g.vertex_count) {
      return;
    }
  }
  TriangleData t;
  source_triangle(g, f, t);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    if (!isfinite(t.x[c]) || !isfinite(t.y[c]) || !isfinite(t.z[c]) || t.z[c] <= 1.e-8f) {
      return;
    }
  }
  if (!isfinite(source_area(t)) || fabsf(source_area(t)) <= 1.e-12f) {
    return;
  }
  const uint32_t a = n * g.vertex_count + t.vertex_id[edge % 3];
  const uint32_t b = n * g.vertex_count + t.vertex_id[(edge + 1) % 3];
  keys[edge] = (static_cast<uint64_t>(min(a, b)) << 32) | max(a, b);
}

__global__ void source_edge_groups(
    GeometryParams g,
    int32_t count,
    const uint64_t* keys,
    const int32_t* ids,
    int32_t* ranges,
    int32_t* adjacency) {
  const int32_t head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head >= count || keys[head] == UINT64_MAX || (head > 0 && keys[head - 1] == keys[head])) {
    return;
  }
  int32_t end = head + 1;
  while (end < count && keys[end] == keys[head]) {
    ++end;
  }
  for (int32_t i = head; i < end; ++i) {
    ranges[static_cast<int64_t>(ids[i]) * 2] = head;
    ranges[static_cast<int64_t>(ids[i]) * 2 + 1] = end;
  }
  if (end - head == 2) {
    TriangleData a, b;
    source_triangle(g, ids[head] / 3, a);
    source_triangle(g, ids[head + 1] / 3, b);
    const int32_t first = min(a.vertex_id[ids[head] % 3], a.vertex_id[(ids[head] + 1) % 3]);
    if ((edge_side(a, ids[head] % 3, first) > 0.0f) !=
        (edge_side(b, ids[head + 1] % 3, first) > 0.0f)) {
      adjacency[ids[head]] = ids[head + 1] / 3;
      adjacency[ids[head + 1]] = ids[head] / 3;
    }
  }
}

std::vector<torch::Tensor> source_build_adjacency(
    const GeometryParams& g,
    const torch::Tensor& vi) {
  const int32_t count = static_cast<int32_t>(vi.numel());
  auto keys = torch::empty({count}, vi.options().dtype(torch::kInt64));
  auto sorted_keys = torch::empty_like(keys);
  auto ids = torch::empty({count}, vi.options());
  auto sorted_ids = torch::empty_like(ids);
  auto ranges = torch::full({count, 2}, -1, vi.options());
  auto adjacency = torch::full({count}, -1, vi.options());
  if (count) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (count + 255) / 256;
    source_edge_keys<<<blocks, 256, 0, stream>>>(
        g, count, reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()), ids.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    size_t bytes = 0;
    C10_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            bytes,
            reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
            reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
            ids.data_ptr<int32_t>(),
            sorted_ids.data_ptr<int32_t>(),
            count,
            0,
            64,
            stream));
    auto scratch = torch::empty({static_cast<int64_t>(bytes)}, vi.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            scratch.data_ptr<uint8_t>(),
            bytes,
            reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
            reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
            ids.data_ptr<int32_t>(),
            sorted_ids.data_ptr<int32_t>(),
            count,
            0,
            64,
            stream));
    source_edge_groups<<<blocks, 256, 0, stream>>>(
        g,
        count,
        reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
        sorted_ids.data_ptr<int32_t>(),
        ranges.data_ptr<int32_t>(),
        adjacency.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {adjacency, ranges, sorted_ids};
}

__device__ int32_t
source_receiver(const SourceParams& p, int32_t face, float ox, float oy, float tx, float ty) {
  int32_t previous = -1;
  float progress = -1.e-5f;
  // A straight segment crosses each edge at most once on a nonfolded sheet.
  for (int64_t step = 0; step <= static_cast<int64_t>(p.geometry.face_count) * 3; ++step) {
    TriangleData t;
    source_triangle(p.geometry, face, t);
    float a[3], b[3];
    if (hard_triangle_coverage(t, tx, ty, source_area(t), b)) {
      return face;
    }
    // Use the same projected origin throughout the walk. Mixing it with the
    // input barycentrics can reverse near-vertex crossing order after rounding.
    source_screen_bary(t, ox, oy, a);
    if (step == 0) {
      // Valid query barycentrics place the origin in this triangle. Projection
      // rounding can put it just outside an edge; retain its known membership.
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        a[c] = fmaxf(a[c], 0.0f);
      }
    }
    source_screen_bary(t, tx, ty, b);
    int32_t next = -1, best_edge = 4;
    float best = 2.0f;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      if (b[c] > 0.0f) {
        continue;
      }
      const int32_t edge = (c + 1) % 3;
      const int32_t neighbor = p.adjacency[static_cast<int64_t>(face) * 3 + edge];
      if (neighbor == previous) {
        continue;
      }
      const float denom = a[c] - b[c];
      const float along = denom > 0.0f ? a[c] / denom : 1.0f;
      if (along < progress - 1.e-5f || along > 1.00001f) {
        continue;
      }
      if (along < best || (along == best && edge < best_edge)) {
        best = along;
        best_edge = edge;
        next = neighbor;
      }
    }
    if (next < 0) {
      return -1;
    }
    previous = face;
    face = next;
    progress = best;
  }
  return -1;
}

__device__ __forceinline__ int32_t
source_tile(const SourceParams& p, int32_t n, int32_t x, int32_t y) {
  const int32_t tw = (p.width + kTileSize - 1) / kTileSize;
  const int32_t th = (p.height + kTileSize - 1) / kTileSize;
  return (n * th + y / kTileSize) * tw + x / kTileSize;
}

__device__ float
source_prefix(const SourceParams& p, int32_t receiver, int32_t tile, float x, float y) {
  TriangleData r;
  source_triangle(p.geometry, receiver, r);
  const uint64_t rank = triangle_order_key(r, receiver);
  float transmittance = 1.0f;
  for (int32_t i = p.offsets[tile]; i < p.offsets[tile + 1]; ++i) {
    const int32_t face = p.faces[i];
    TriangleData t;
    source_triangle(p.geometry, face, t);
    if (triangle_order_key(t, face) >= rank) {
      break;
    }
    float sb[3], pb[3], iz, a;
    if (hard_triangle_coverage(t, x, y, source_area(t), sb)) {
      transmittance *= 1.0f - source_alpha(t, sb, pb, iz, a);
      if (transmittance == 0.0f) {
        break;
      }
    }
  }
  return transmittance;
}

__device__ bool
source_query(const SourceParams& p, int32_t query, int32_t& face, float& x, float& y, float& z) {
  const int32_t n = query / p.query_count;
  const int32_t local = p.queries[query];
  if (local < 0 || local >= p.geometry.face_count) {
    return false;
  }
  face = n * p.geometry.face_count + local;
  if (p.edge_ranges[static_cast<int64_t>(face) * 6] < 0) {
    return false;
  }
  const float* beta = p.bary + static_cast<int64_t>(query) * 3;
  float sum = 0.0f;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    if (!isfinite(beta[c]) || beta[c] < 0.0f) {
      return false;
    }
    sum += beta[c];
  }
  if (fabsf(sum - 1.0f) >= 1.e-5f) {
    return false;
  }
  TriangleData t;
  source_triangle(p.geometry, face, t);
  x = y = z = 0.0f;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float wz = beta[c] * t.z[c];
    z += wz;
    x += wz * t.x[c];
    y += wz * t.y[c];
  }
  x /= z;
  y /= z;
  return isfinite(x) && isfinite(y) && isfinite(z);
}

__global__ void source_stencil_kernel(
    SourceParams p,
    int32_t total,
    int32_t* indices,
    float* weights,
    float* uv,
    float* depth,
    int32_t* receivers,
    float* visibility) {
  const int32_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= total) {
    return;
  }
  int32_t face;
  float x, y, z;
  if (!source_query(p, query, face, x, y, z)) {
    return;
  }
  uv[query * 2] = 2.0f * (x + 0.5f) / p.width - 1.0f;
  uv[query * 2 + 1] = 2.0f * (y + 0.5f) / p.height - 1.0f;
  depth[query] = z;
  if (x < -1.0f || y < -1.0f || x >= p.width || y >= p.height) {
    return;
  }
  const int32_t x0 = static_cast<int32_t>(floorf(x)), y0 = static_cast<int32_t>(floorf(y));
  const float fx = x - x0, fy = y - y0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int32_t tx = x0 + (j & 1), ty = y0 + (j >> 1);
    if (tx < 0 || ty < 0 || tx >= p.width || ty >= p.height) {
      continue;
    }
    const int64_t slot = static_cast<int64_t>(query) * 4 + j;
    indices[slot] = ty * p.width + tx;
    const int32_t receiver = source_receiver(p, face, x, y, tx, ty);
    receivers[slot] = receiver;
    if (receiver >= 0) {
      const float trans =
          source_prefix(p, receiver, source_tile(p, query / p.query_count, tx, ty), tx, ty);
      visibility[slot] = trans;
      weights[slot] = trans * ((j & 1) ? fx : 1.0f - fx) * ((j >> 1) ? fy : 1.0f - fy);
    }
  }
}

__device__ void source_material_backward(
    const SourceParams& p,
    int32_t receiver,
    int32_t tile,
    int32_t x,
    int32_t y,
    float grad,
    float* gv,
    float* go) {
  TriangleData r;
  source_triangle(p.geometry, receiver, r);
  const uint64_t rank = triangle_order_key(r, receiver);
  ScaledTransmittance nonzero;
  int32_t zeros = 0;
  for (int32_t i = p.offsets[tile]; i < p.offsets[tile + 1]; ++i) {
    const int32_t face = p.faces[i];
    TriangleData t;
    source_triangle(p.geometry, face, t);
    if (triangle_order_key(t, face) >= rank) {
      break;
    }
    float sb[3], pb[3], iz, a;
    if (hard_triangle_coverage(t, x, y, source_area(t), sb)) {
      const float factor = 1.0f - source_alpha(t, sb, pb, iz, a);
      if (factor == 0.0f) {
        ++zeros;
      } else {
        nonzero.multiply(factor);
      }
    }
  }
  if (zeros > 1) {
    return;
  }
  const int32_t n = receiver / p.geometry.face_count;
  for (int32_t i = p.offsets[tile]; i < p.offsets[tile + 1]; ++i) {
    const int32_t face = p.faces[i];
    TriangleData t;
    source_triangle(p.geometry, face, t);
    if (triangle_order_key(t, face) >= rank) {
      break;
    }
    float sb[3], pb[3], iz, a;
    if (!hard_triangle_coverage(t, x, y, source_area(t), sb)) {
      continue;
    }
    const float factor = 1.0f - source_alpha(t, sb, pb, iz, a);
    if (a < 0.0f || a > 1.0f || (zeros == 1 && factor != 0.0f)) {
      continue;
    }
    const float except =
        factor == 0.0f ? nonzero.value() : ldexpf(nonzero.mantissa / factor, -nonzero.exponent);
    const float ga = -grad * except;
    float gb[3], gx[3] = {}, gy[3] = {}, gz[3] = {};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int64_t vertex = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[c];
      atomicAdd(go + vertex, ga * (c == 0 ? 1.0f - pb[1] - pb[2] : pb[c]));
      gb[c] = ga * (t.opacity[c] - t.opacity[0]);
    }
    perspective_barycentric_backward(t, x, y, sb, pb, iz, gb, gx, gy, gz);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int64_t vertex = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[c];
      atomicAdd(gv + vertex * 3, gx[c]);
      atomicAdd(gv + vertex * 3 + 1, gy[c]);
      atomicAdd(gv + vertex * 3 + 2, gz[c]);
    }
  }
}

__device__ float source_boundary_limit(
    const SourceParams& p,
    int32_t receiver,
    int32_t tile,
    int32_t x,
    int32_t y,
    int32_t first,
    int32_t second,
    int32_t begin,
    int32_t end,
    float alpha,
    bool plus) {
  if (receiver < 0) {
    return 0.0f;
  }
  TriangleData r;
  source_triangle(p.geometry, receiver, r);
  const uint64_t rank = triangle_order_key(r, receiver);
  float trans = 1.0f;
  for (int32_t i = p.offsets[tile]; i < p.offsets[tile + 1]; ++i) {
    const int32_t face = p.faces[i];
    TriangleData t;
    source_triangle(p.geometry, face, t);
    if (triangle_order_key(t, face) >= rank) {
      break;
    }
    bool has_first = false, has_second = false;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      has_first |= t.vertex_id[c] == first;
      has_second |= t.vertex_id[c] == second;
    }
    if (has_first && has_second) {
      continue;
    }
    float sb[3], pb[3], iz, a;
    if (hard_triangle_coverage(t, x, y, source_area(t), sb)) {
      trans *= 1.0f - source_alpha(t, sb, pb, iz, a);
    }
  }
  for (int32_t i = begin; i < end; ++i) {
    const int32_t edge = p.edge_ids[i];
    TriangleData t;
    source_triangle(p.geometry, edge / 3, t);
    if (triangle_order_key(t, edge / 3) < rank &&
        ((edge_side(t, edge % 3, first) > 0.0f) == plus)) {
      trans *= 1.0f - alpha;
    }
  }
  return trans;
}

__device__ void source_edge_backward(
    const SourceParams& p,
    int32_t query,
    int32_t query_face,
    float qx,
    float qy,
    int32_t tile,
    int32_t x,
    int32_t y,
    int32_t actual_receiver,
    float grad,
    float* gv) {
  const int32_t n = query_face / p.geometry.face_count;
  for (int32_t i = p.offsets[tile]; i < p.offsets[tile + 1]; ++i) {
    const int32_t face = p.faces[i];
#pragma unroll
    for (int e = 0; e < 3; ++e) {
      const int32_t edge = face * 3 + e;
      const int32_t begin = p.edge_ranges[static_cast<int64_t>(edge) * 2];
      if (begin < 0 || p.edge_ids[begin] != edge || p.adjacency[edge] >= 0) {
        continue;
      }
      const int32_t end = p.edge_ranges[static_cast<int64_t>(edge) * 2 + 1];
      TriangleData t;
      source_triangle(p.geometry, face, t);
      int32_t a = e, b = (e + 1) % 3;
      if (t.vertex_id[a] > t.vertex_id[b]) {
        const int32_t tmp = a;
        a = b;
        b = tmp;
      }
#pragma unroll
      for (int axis = 0; axis < 2; ++axis) {
        BoundaryPixelPair pair;
        if (!boundary_pixel_pair(
                t, a, b, axis, axis == 0 ? y : x, axis == 0 ? p.width : p.height, pair)) {
          continue;
        }
        const int32_t along = axis == 0 ? x : y;
        if (along != pair.low && along != pair.low + 1) {
          continue;
        }
        const float bx = t.x[a] + pair.t * (t.x[b] - t.x[a]);
        const float by = t.y[a] + pair.t * (t.y[b] - t.y[a]);
        // Sample the one-sided receiver limits, not a soft visibility test.
        const float epsilon = 1.e-4f;
        const float ex = axis == 0 ? pair.sign * epsilon : 0.0f;
        const float ey = axis == 1 ? pair.sign * epsilon : 0.0f;
        int32_t rp = source_receiver(p, query_face, qx, qy, bx - ex, by - ey);
        int32_t rm = source_receiver(p, query_face, qx, qy, bx + ex, by + ey);
        bool receiver_boundary = false;
        for (int32_t k = begin; k < end; ++k) {
          receiver_boundary |= p.edge_ids[k] / 3 == rp || p.edge_ids[k] / 3 == rm;
        }
        if (!receiver_boundary) {
          // A receiver can cross its own ordinary tessellation edge between
          // the boundary crossing and this pixel. Keep the pixel's owner.
          rp = rm = actual_receiver;
        }
        const float ba = (1.0f - pair.t) / t.z[a], bb = pair.t / t.z[b];
        const float alpha =
            fminf(fmaxf((ba * t.opacity[a] + bb * t.opacity[b]) / (ba + bb), 0.0f), 1.0f);
        if (alpha == 0.0f && rp == rm) {
          continue;
        }
        const float plus = source_boundary_limit(
            p, rp, tile, x, y, t.vertex_id[a], t.vertex_id[b], begin, end, alpha, true);
        const float minus = source_boundary_limit(
            p, rm, tile, x, y, t.vertex_id[a], t.vertex_id[b], begin, end, alpha, false);
        const float force = 0.5f * pair.sign * grad * (plus - minus);
        const int64_t va = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[a];
        const int64_t vb = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[b];
        atomicAdd(gv + va * 3 + axis, (1.0f - pair.t) * force);
        atomicAdd(gv + vb * 3 + axis, pair.t * force);
      }
    }
  }
}

__global__ void source_stencil_backward_kernel(
    SourceParams p,
    int32_t total,
    const int32_t* indices,
    const int32_t* receivers,
    const float* visibility,
    const float* gw,
    const float* guv,
    const float* gz,
    bool edges,
    float* gv,
    float* gb,
    float* go) {
  const int32_t query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= total) {
    return;
  }
  int32_t face;
  float x, y, z;
  if (!source_query(p, query, face, x, y, z)) {
    return;
  }
  const int32_t n = query / p.query_count;
  float gx = guv[query * 2] * 2.0f / p.width;
  float gy = guv[query * 2 + 1] * 2.0f / p.height;
  const float fx = x - floorf(x), fy = y - floorf(y);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t slot = static_cast<int64_t>(query) * 4 + j;
    const int32_t index = indices[slot];
    if (index < 0 || gw[slot] == 0.0f) {
      continue;
    }
    const float wx = (j & 1) ? fx : 1.0f - fx;
    const float wy = (j >> 1) ? fy : 1.0f - fy;
    gx += gw[slot] * visibility[slot] * ((j & 1) ? 1.0f : -1.0f) * wy;
    gy += gw[slot] * visibility[slot] * ((j >> 1) ? 1.0f : -1.0f) * wx;
    const int32_t tx = index % p.width, ty = index / p.width;
    const int32_t tile = source_tile(p, n, tx, ty);
    const float grad = gw[slot] * wx * wy;
    if (grad != 0.0f) {
      if (receivers[slot] >= 0) {
        source_material_backward(p, receivers[slot], tile, tx, ty, grad, gv, go);
      }
      if (edges) {
        source_edge_backward(p, query, face, x, y, tile, tx, ty, receivers[slot], grad, gv);
      }
    }
  }
  TriangleData t;
  source_triangle(p.geometry, face, t);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int64_t vertex = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[c];
    const float beta = p.bary[static_cast<int64_t>(query) * 3 + c];
    const float projection = ((t.x[c] - x) * gx + (t.y[c] - y) * gy) / z;
    atomicAdd(gv + vertex * 3, beta * t.z[c] * gx / z);
    atomicAdd(gv + vertex * 3 + 1, beta * t.z[c] * gy / z);
    atomicAdd(gv + vertex * 3 + 2, beta * (projection + gz[query]));
    gb[static_cast<int64_t>(query) * 3 + c] = t.z[c] * (projection + gz[query]);
  }
}

struct FeatureShape {
  int64_t batch, channels, height, width, queries;
  int64_t stride[4];
};

template <typename scalar_t>
__global__ void source_sample_kernel(
    FeatureShape s,
    const scalar_t* f,
    const int32_t* indices,
    const float* weights,
    scalar_t* samples,
    float* confidence) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t nq = s.batch * s.queries;
  if (index < nq) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int32_t tap = indices[index * 4 + j];
      if (tap >= 0 && tap < s.height * s.width) {
        sum += weights[index * 4 + j];
      }
    }
    confidence[index] = sum;
  }
  if (index >= nq * s.channels) {
    return;
  }
  const int64_t q = index % s.queries;
  const int64_t c = (index / s.queries) % s.channels;
  const int64_t n = index / (s.queries * s.channels);
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t slot = (n * s.queries + q) * 4 + j;
    const int32_t tap = indices[slot];
    if (tap >= 0 && tap < s.height * s.width) {
      const int64_t src = n * s.stride[0] + c * s.stride[1] + (tap / s.width) * s.stride[2] +
          (tap % s.width) * s.stride[3];
      sum += weights[slot] * static_cast<float>(f[src]);
    }
  }
  samples[index] = static_cast<scalar_t>(sum);
}

template <typename scalar_t>
__global__ void source_sample_image_backward_kernel(
    FeatureShape s,
    const int32_t* indices,
    const float* weights,
    const scalar_t* gs,
    float* gf) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= s.batch * s.channels * s.queries) {
    return;
  }
  const int64_t q = index % s.queries;
  const int64_t c = (index / s.queries) % s.channels;
  const int64_t n = index / (s.queries * s.channels);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t slot = (n * s.queries + q) * 4 + j;
    const int32_t tap = indices[slot];
    if (tap >= 0 && tap < s.height * s.width) {
      atomicAdd(
          gf + (n * s.channels + c) * s.height * s.width + tap,
          weights[slot] * static_cast<float>(gs[index]));
    }
  }
}

template <typename scalar_t>
__global__ void source_sample_weight_backward_kernel(
    FeatureShape s,
    const scalar_t* f,
    const int32_t* indices,
    const scalar_t* gs,
    const float* gc,
    float* gw) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= s.batch * s.queries * 4) {
    return;
  }
  const int64_t query = index / 4;
  const int64_t n = query / s.queries, q = query % s.queries;
  const int32_t tap = indices[index];
  float sum = 0.0f;
  if (tap >= 0 && tap < s.height * s.width) {
    sum = gc[query];
    const int64_t base =
        n * s.stride[0] + (tap / s.width) * s.stride[2] + (tap % s.width) * s.stride[3];
    for (int64_t c = 0; c < s.channels; ++c) {
      sum += static_cast<float>(gs[(n * s.channels + c) * s.queries + q]) *
          static_cast<float>(f[base + c * s.stride[1]]);
    }
  }
  gw[index] = sum;
}

GeometryParams
source_geometry(const torch::Tensor& v, const torch::Tensor& vi, const torch::Tensor& opacity) {
  return {
      v.data_ptr<float>(),
      vi.data_ptr<int32_t>(),
      nullptr,
      nullptr,
      opacity.data_ptr<float>(),
      static_cast<int32_t>(v.size(1)),
      static_cast<int32_t>(vi.size(1))};
}

SourceParams source_params(
    const GeometryParams& g,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& adjacency,
    const torch::Tensor& ranges,
    const torch::Tensor& edges,
    const torch::Tensor& offsets,
    const torch::Tensor& faces,
    int64_t h,
    int64_t w) {
  return {
      g,
      qf.data_ptr<int32_t>(),
      qb.data_ptr<float>(),
      adjacency.data_ptr<int32_t>(),
      ranges.data_ptr<int32_t>(),
      edges.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>(),
      faces.data_ptr<int32_t>(),
      static_cast<int32_t>(qf.size(1)),
      static_cast<int32_t>(h),
      static_cast<int32_t>(w)};
}

void source_check(const torch::Tensor& t, at::ScalarType dtype, const torch::Device& device) {
  TORCH_CHECK(
      t.is_cuda() && t.device() == device && t.scalar_type() == dtype && t.is_contiguous(),
      "source features: invalid device, dtype, or contiguous layout");
}

FeatureShape
source_feature_shape(const torch::Tensor& f, const torch::Tensor& idx, const torch::Tensor& w) {
  TORCH_CHECK(
      f.is_cuda() && f.dim() == 4 && idx.dim() == 3 && idx.size(2) == 4 &&
          idx.size(0) == f.size(0) && w.sizes() == idx.sizes(),
      "source features: invalid sampler shapes");
  TORCH_CHECK(
      f.scalar_type() == at::kFloat || f.scalar_type() == at::kHalf ||
          f.scalar_type() == at::kBFloat16,
      "source features: expected fp16/bf16/fp32 image");
  source_check(idx, at::kInt, f.device());
  source_check(w, at::kFloat, f.device());
  TORCH_CHECK(
      f.size(2) > 0 && f.size(3) > 0 && f.size(2) * f.size(3) < INT32_MAX,
      "source features: invalid image size");
  FeatureShape shape{f.size(0), f.size(1), f.size(2), f.size(3), idx.size(1), {}};
  for (int i = 0; i < 4; ++i) {
    shape.stride[i] = f.stride(i);
  }
  return shape;
}

} // namespace

std::vector<torch::Tensor> source_stencil_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& opacity,
    int64_t height,
    int64_t width) {
  source_check(v, at::kFloat, v.device());
  source_check(vi, at::kInt, v.device());
  source_check(qf, at::kInt, v.device());
  source_check(qb, at::kFloat, v.device());
  source_check(opacity, at::kFloat, v.device());
  TORCH_CHECK(
      v.dim() == 3 && v.size(2) == 3 && vi.dim() == 3 && vi.size(0) == v.size(0) &&
          vi.size(2) == 3 && qf.dim() == 2 && qf.size(0) == v.size(0) && qb.dim() == 3 &&
          qb.size(0) == qf.size(0) && qb.size(1) == qf.size(1) && qb.size(2) == 3 &&
          opacity.dim() == 2 && opacity.size(0) == v.size(0) && opacity.size(1) == v.size(1),
      "source features: invalid stencil shapes");
  TORCH_CHECK(
      height > 0 && width > 0 && height < INT32_MAX && width < INT32_MAX &&
          height * width < INT32_MAX && vi.numel() < INT32_MAX && v.numel() < INT32_MAX &&
          qf.numel() < INT32_MAX / 4,
      "source features: dimensions exceed int32 indexing");
  const c10::cuda::CUDAGuard guard(v.device());
  const auto g = source_geometry(v, vi, opacity);
  const auto adjacency = source_build_adjacency(g, vi);
  const auto bins =
      build_tile_bins(v, vi, static_cast<int32_t>(height), static_cast<int32_t>(width), 1);
  const int64_t n = qf.size(0), q = qf.size(1);
  auto indices = torch::full({n, q, 4}, -1, vi.options());
  auto weights = torch::zeros({n, q, 4}, v.options());
  auto uv = torch::zeros({n, q, 2}, v.options());
  auto depth = torch::zeros({n, q}, v.options());
  auto receivers = torch::full_like(indices, -1);
  auto visibility = torch::zeros_like(weights);
  if (n * q != 0) {
    auto p = source_params(
        g,
        qf,
        qb,
        adjacency[0],
        adjacency[1],
        adjacency[2],
        bins.offsets,
        bins.face_ids,
        height,
        width);
    source_stencil_kernel<<<(n * q + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        p,
        static_cast<int32_t>(n * q),
        indices.data_ptr<int32_t>(),
        weights.data_ptr<float>(),
        uv.data_ptr<float>(),
        depth.data_ptr<float>(),
        receivers.data_ptr<int32_t>(),
        visibility.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {
      indices,
      weights,
      uv,
      depth,
      receivers,
      visibility,
      bins.offsets,
      bins.face_ids,
      adjacency[0],
      adjacency[1],
      adjacency[2]};
}

std::vector<torch::Tensor> source_stencil_backward(
    const torch::TensorList& s,
    const torch::TensorList& grads,
    int64_t height,
    int64_t width,
    bool edge_gradients) {
  TORCH_CHECK(s.size() == 16 && grads.size() == 3, "source features: invalid saved state");
  const auto& v = s[0];
  const c10::cuda::CUDAGuard guard(v.device());
  auto gv = torch::zeros_like(v), gb = torch::zeros_like(s[3]), go = torch::zeros_like(s[4]);
  const auto total = s[2].numel();
  if (total) {
    const auto g = source_geometry(v, s[1], s[4]);
    const auto p = source_params(g, s[2], s[3], s[13], s[14], s[15], s[11], s[12], height, width);
    source_stencil_backward_kernel<<<
        (total + 127) / 128,
        128,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        p,
        static_cast<int32_t>(total),
        s[5].data_ptr<int32_t>(),
        s[9].data_ptr<int32_t>(),
        s[10].data_ptr<float>(),
        grads[0].data_ptr<float>(),
        grads[1].data_ptr<float>(),
        grads[2].data_ptr<float>(),
        edge_gradients,
        gv.data_ptr<float>(),
        gb.data_ptr<float>(),
        go.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {gv, gb, go};
}

std::vector<torch::Tensor> source_sample_forward(
    const torch::Tensor& f,
    const torch::Tensor& idx,
    const torch::Tensor& weights) {
  const auto s = source_feature_shape(f, idx, weights);
  const c10::cuda::CUDAGuard guard(f.device());
  auto out = torch::empty({s.batch, s.channels, s.queries}, f.options());
  auto confidence = torch::empty({s.batch, 1, s.queries}, weights.options());
  const int64_t threads = s.batch * std::max<int64_t>(s.channels, 1) * s.queries;
  if (threads) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, f.scalar_type(), "source_sample_forward", [&] {
          source_sample_kernel<scalar_t>
              <<<(threads + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  s,
                  f.data_ptr<scalar_t>(),
                  idx.data_ptr<int32_t>(),
                  weights.data_ptr<float>(),
                  out.data_ptr<scalar_t>(),
                  confidence.data_ptr<float>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out, confidence};
}

std::vector<torch::Tensor> source_sample_backward(
    const torch::Tensor& f,
    const torch::Tensor& idx,
    const torch::Tensor& weights,
    const torch::Tensor& gs,
    const torch::Tensor& gc) {
  const auto s = source_feature_shape(f, idx, weights);
  const c10::cuda::CUDAGuard guard(f.device());
  auto gf = torch::zeros(f.sizes(), f.options().dtype(at::kFloat));
  auto gw = torch::empty_like(weights);
  const int64_t threads = s.batch * s.channels * s.queries;
  const int64_t slots = idx.numel();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, f.scalar_type(), "source_sample_backward", [&] {
        if (threads) {
          source_sample_image_backward_kernel<scalar_t>
              <<<(threads + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  s,
                  idx.data_ptr<int32_t>(),
                  weights.data_ptr<float>(),
                  gs.data_ptr<scalar_t>(),
                  gf.data_ptr<float>());
        }
        if (slots) {
          source_sample_weight_backward_kernel<scalar_t>
              <<<(slots + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  s,
                  f.data_ptr<scalar_t>(),
                  idx.data_ptr<int32_t>(),
                  gs.data_ptr<scalar_t>(),
                  gc.data_ptr<float>(),
                  gw.data_ptr<float>());
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gf.to(f.scalar_type()), gw};
}

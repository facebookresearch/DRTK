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
  return math::cross(
      float2{t.x[1] - t.x[0], t.y[1] - t.y[0]}, float2{t.x[2] - t.x[0], t.y[2] - t.y[0]});
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

struct SourceMaterialEvaluator {
  static constexpr bool shaded = false;
  static constexpr bool shared_grads = false;
};

template <typename Fragment, bool SharedGrads = false>
struct SourceAppearanceEvaluator : Fragment {
  static constexpr bool shaded = true;
  static constexpr bool shared_grads = SharedGrads;
};

__device__ void source_load_uv(const GeometryParams& g, int32_t n, TriangleData& t) {
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int64_t vertex = static_cast<int64_t>(n) * g.vertex_count + t.vertex_id[c];
    t.u[c] = g.uv[vertex * 2];
    t.v[c] = g.uv[vertex * 2 + 1];
    if (g.reference_uv != nullptr) {
      t.reference_u[c] = g.reference_uv[vertex * 2];
      t.reference_v[c] = g.reference_uv[vertex * 2 + 1];
    }
  }
}

template <typename Evaluator>
__device__ float source_combined_alpha(
    const GeometryParams& g,
    const Evaluator& evaluator,
    int32_t n,
    TriangleData& t,
    const float* bary,
    float material,
    float& appearance) {
  material = fminf(fmaxf(material, 0.0f), 1.0f);
  appearance = 1.0f;
  if constexpr (Evaluator::shaded) {
    source_load_uv(g, n, t);
    float rgb[3];
    evaluator.evaluate(n, t, bary, rgb, appearance);
  }
  return fminf(material * appearance, kAlphaMax);
}

template <typename Evaluator>
__device__ float source_alpha(
    const GeometryParams& g,
    const Evaluator& evaluator,
    int32_t face,
    TriangleData& t,
    const float* screen,
    float* bary,
    float& inverse_depth,
    float& unclamped,
    float& appearance) {
  inverse_depth = 0.0f;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    bary[c] = screen[c] / t.z[c];
    inverse_depth += bary[c];
  }
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    bary[c] /= inverse_depth;
  }
  unclamped = interpolate_material_opacity(t, bary);
  return source_combined_alpha(g, evaluator, face / g.face_count, t, bary, unclamped, appearance);
}

C10_LAUNCH_BOUNDS_1(256)
__global__ void source_edge_keys(GeometryParams g, int32_t count, uint64_t* keys, int32_t* ids) {
  CUDA_KERNEL_LOOP_TYPE(edge, count, int32_t) {
    keys[edge] = UINT64_MAX;
    ids[edge] = edge;
    const int32_t f = edge / 3, n = f / g.face_count;
    bool valid = true;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int32_t id = g.faces[static_cast<int64_t>(f) * 3 + c];
      valid = valid && id >= 0 && id < g.vertex_count;
    }
    if (!valid)
      continue;
    TriangleData t;
    source_triangle(g, f, t);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      valid = valid && isfinite(t.x[c]) && isfinite(t.y[c]) && isfinite(t.z[c]) && t.z[c] > 1.e-8f;
    }
    if (!valid)
      continue;
    if (!isfinite(source_area(t)) || fabsf(source_area(t)) <= 1.e-12f) {
      continue;
    }
    const uint32_t a = n * g.vertex_count + t.vertex_id[edge % 3];
    const uint32_t b = n * g.vertex_count + t.vertex_id[(edge + 1) % 3];
    keys[edge] = (static_cast<uint64_t>(min(a, b)) << 32) | max(a, b);
  }
}

C10_LAUNCH_BOUNDS_1(256)
__global__ void source_edge_groups(
    GeometryParams g,
    int32_t count,
    const uint64_t* keys,
    const int32_t* ids,
    int32_t* ranges,
    int32_t* adjacency) {
  CUDA_KERNEL_LOOP_TYPE(head, count, int32_t) {
    if (keys[head] == UINT64_MAX || (head > 0 && keys[head - 1] == keys[head])) {
      continue;
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
    const int blocks = GET_BLOCKS(count, 256);
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

template <typename Evaluator>
__device__ float source_prefix(
    const SourceParams& p,
    const Evaluator& evaluator,
    int32_t receiver,
    int32_t tile,
    float x,
    float y) {
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
    float sb[3], pb[3], iz, a, appearance;
    if (hard_triangle_coverage(t, x, y, source_area(t), sb)) {
      transmittance *=
          1.0f - source_alpha(p.geometry, evaluator, face, t, sb, pb, iz, a, appearance);
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

template <typename Evaluator>
C10_LAUNCH_BOUNDS_1(128)
__global__ void source_stencil_kernel(
    SourceParams p,
    const Evaluator evaluator,
    int32_t total,
    int32_t* indices,
    float* weights,
    float* uv,
    float* depth,
    int32_t* receivers,
    float* visibility) {
  CUDA_KERNEL_LOOP_TYPE(query, total, int32_t) {
    int32_t face;
    float x, y, z;
    if (!source_query(p, query, face, x, y, z)) {
      continue;
    }
    uv[query * 2] = 2.0f * (x + 0.5f) / p.width - 1.0f;
    uv[query * 2 + 1] = 2.0f * (y + 0.5f) / p.height - 1.0f;
    depth[query] = z;
    if (x < -1.0f || y < -1.0f || x >= p.width || y >= p.height) {
      continue;
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
        const float trans = source_prefix(
            p, evaluator, receiver, source_tile(p, query / p.query_count, tx, ty), tx, ty);
        visibility[slot] = trans;
        weights[slot] = trans * ((j & 1) ? fx : 1.0f - fx) * ((j >> 1) ? fy : 1.0f - fy);
      }
    }
  }
}

template <typename Evaluator>
__device__ void source_material_backward(
    const SourceParams& p,
    const Evaluator& evaluator,
    int32_t receiver,
    int32_t tile,
    int32_t x,
    int32_t y,
    float grad,
    float* gv,
    float* go,
    float* gu,
    float* gu1) {
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
    float sb[3], pb[3], iz, a, appearance;
    if (hard_triangle_coverage(t, x, y, source_area(t), sb)) {
      const float factor =
          1.0f - source_alpha(p.geometry, evaluator, face, t, sb, pb, iz, a, appearance);
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
    float sb[3], pb[3], iz, a, appearance;
    if (!hard_triangle_coverage(t, x, y, source_area(t), sb)) {
      continue;
    }
    const float alpha = source_alpha(p.geometry, evaluator, face, t, sb, pb, iz, a, appearance);
    const float factor = 1.0f - alpha;
    // Unshaded material outside the clamp has no derivative. With an
    // appearance model, a saturated material can still transmit its gradients.
    if ((zeros == 1 && factor != 0.0f) || (!Evaluator::shaded && (a < 0.0f || a > 1.0f))) {
      continue;
    }
    const float except =
        factor == 0.0f ? nonzero.value() : ldexpf(nonzero.mantissa / factor, -nonzero.exponent);
    // A small positive alpha can round 1-alpha to one without reaching the
    // zero-alpha ReLU endpoint. Preserve its opacity derivative.
    const float ga = Evaluator::shaded && alpha == 0.0f ? 0.0f : -grad * except;
    const float material = fminf(fmaxf(a, 0.0f), 1.0f);
    const float gm = a >= 0.0f && a <= 1.0f ? ga * appearance : 0.0f;
    FragmentAttributeGrads attr{};
    if constexpr (Evaluator::shaded) {
      const float rgb_grad[3] = {};
      evaluator.backward(n, t, pb, rgb_grad, ga * material, appearance, attr);
    }
    float gx[3] = {}, gy[3] = {}, gz[3] = {};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int64_t vertex = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[c];
      atomicAdd(go + vertex, gm * (c == 0 ? 1.0f - pb[1] - pb[2] : pb[c]));
      attr.bary[c] += gm * (t.opacity[c] - t.opacity[0]);
      if constexpr (Evaluator::shaded) {
        atomicAdd(gu + vertex * 2, attr.u[c]);
        atomicAdd(gu + vertex * 2 + 1, attr.v[c]);
        if (gu1 != nullptr) {
          atomicAdd(gu1 + vertex * 2, attr.reference_u[c]);
          atomicAdd(gu1 + vertex * 2 + 1, attr.reference_v[c]);
        }
      }
    }
    perspective_barycentric_backward(t, x, y, sb, pb, iz, attr.bary, gx, gy, gz);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const int64_t vertex = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[c];
      atomicAdd(gv + vertex * 3, gx[c]);
      atomicAdd(gv + vertex * 3 + 1, gy[c]);
      atomicAdd(gv + vertex * 3 + 2, gz[c]);
    }
  }
}

template <typename Evaluator>
__device__ float source_boundary_limit(
    const SourceParams& p,
    const Evaluator& evaluator,
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
    float sb[3], pb[3], iz, a, appearance;
    if (hard_triangle_coverage(t, x, y, source_area(t), sb)) {
      trans *= 1.0f - source_alpha(p.geometry, evaluator, face, t, sb, pb, iz, a, appearance);
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

template <typename Evaluator>
__device__ void source_edge_backward(
    const SourceParams& p,
    const Evaluator& evaluator,
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
        float pb[3] = {};
        pb[a] = ba / (ba + bb);
        pb[b] = bb / (ba + bb);
        float appearance;
        const float material = t.opacity[a] + pb[b] * (t.opacity[b] - t.opacity[a]);
        const float alpha =
            source_combined_alpha(p.geometry, evaluator, n, t, pb, material, appearance);
        if (alpha == 0.0f && rp == rm) {
          continue;
        }
        const float plus = source_boundary_limit(
            p, evaluator, rp, tile, x, y, t.vertex_id[a], t.vertex_id[b], begin, end, alpha, true);
        const float minus = source_boundary_limit(
            p, evaluator, rm, tile, x, y, t.vertex_id[a], t.vertex_id[b], begin, end, alpha, false);
        const float force = 0.5f * pair.sign * grad * (plus - minus);
        const int64_t va = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[a];
        const int64_t vb = static_cast<int64_t>(n) * p.geometry.vertex_count + t.vertex_id[b];
        atomicAdd(gv + va * 3 + axis, (1.0f - pair.t) * force);
        atomicAdd(gv + vb * 3 + axis, pair.t * force);
      }
    }
  }
}

template <typename Evaluator>
__device__ void source_stencil_backward_query(
    SourceParams p,
    const Evaluator& evaluator,
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
    float* go,
    float* gu,
    float* gu1,
    int32_t query) {
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
        source_material_backward(
            p, evaluator, receivers[slot], tile, tx, ty, grad, gv, go, gu, gu1);
      }
      if (edges) {
        source_edge_backward(
            p, evaluator, query, face, x, y, tile, tx, ty, receivers[slot], grad, gv);
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

template <typename Evaluator>
C10_LAUNCH_BOUNDS_1(128)
__global__ void source_stencil_backward_kernel(
    SourceParams p,
    Evaluator evaluator,
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
    float* go,
    float* gu,
    float* gu1) {
  __shared__ float shared_weights[kMaxMlpWeightParams];
  __shared__ float shared_biases[kMaxMlpBiasParams];
  if constexpr (Evaluator::shared_grads) {
    for (int i = threadIdx.x; i < evaluator.shared_weight_grad_count(); i += blockDim.x) {
      shared_weights[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < evaluator.shared_bias_grad_count(); i += blockDim.x) {
      shared_biases[i] = 0.0f;
    }
    __syncthreads();
    evaluator.bind_shared_mlp_grads(shared_weights, shared_biases);
  }
  CUDA_KERNEL_LOOP_TYPE(query, total, int32_t) {
    source_stencil_backward_query(
        p,
        evaluator,
        total,
        indices,
        receivers,
        visibility,
        gw,
        guv,
        gz,
        edges,
        gv,
        gb,
        go,
        gu,
        gu1,
        query);
  }
  if constexpr (Evaluator::shared_grads) {
    __syncthreads();
    evaluator.flush_shared_mlp_grads(threadIdx.x, blockDim.x);
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void source_sample_kernel(
    const index_t nthreads,
    TensorInfoCompact<scalar_t, index_t, 4> features,
    TensorInfoCompact<int32_t, index_t, 3> indices,
    TensorInfoCompact<float, index_t, 3> weights,
    TensorInfoCompact<scalar_t, index_t, 3> samples,
    TensorInfoCompact<float, index_t, 3> confidence) {
  const index_t channels = features.sizes[1], height = features.sizes[2], width = features.sizes[3];
  const index_t queries = indices.sizes[1];
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    if (index < indices.sizes[0] * queries) {
      const index_t n = index / queries, q = index % queries;
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int32_t tap = indices[n][q][j];
        if (tap >= 0 && tap < height * width)
          sum += weights[n][q][j];
      }
      confidence[n][0][q] = sum;
    }
    if (index >= indices.sizes[0] * channels * queries)
      continue;
    const index_t q = index % queries, c = (index / queries) % channels;
    const index_t n = index / (queries * channels);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int32_t tap = indices[n][q][j];
      if (tap >= 0 && tap < height * width)
        sum += weights[n][q][j] * static_cast<float>(features[n][c][tap / width][tap % width]);
    }
    samples[n][c][q] = static_cast<scalar_t>(sum);
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void source_sample_image_backward_kernel(
    const index_t nthreads,
    TensorInfoCompact<int32_t, index_t, 3> indices,
    TensorInfoCompact<float, index_t, 3> weights,
    TensorInfoCompact<scalar_t, index_t, 3> grad_samples,
    TensorInfoCompact<float, index_t, 4> grad_features) {
  const index_t channels = grad_features.sizes[1], height = grad_features.sizes[2],
                width = grad_features.sizes[3];
  const index_t queries = indices.sizes[1];
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t q = index % queries, c = (index / queries) % channels;
    const index_t n = index / (queries * channels);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int32_t tap = indices[n][q][j];
      if (tap >= 0 && tap < height * width)
        atomicAdd(
            &grad_features[n][c][tap / width][tap % width],
            weights[n][q][j] * static_cast<float>(grad_samples[n][c][q]));
    }
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void source_sample_weight_backward_kernel(
    const index_t nthreads,
    TensorInfoCompact<scalar_t, index_t, 4> features,
    TensorInfoCompact<int32_t, index_t, 3> indices,
    TensorInfoCompact<scalar_t, index_t, 3> grad_samples,
    TensorInfoCompact<float, index_t, 3> grad_confidence,
    TensorInfoCompact<float, index_t, 3> grad_weights) {
  const index_t channels = features.sizes[1], height = features.sizes[2], width = features.sizes[3];
  const index_t queries = indices.sizes[1];
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t n = index / (queries * 4), q = (index / 4) % queries, j = index % 4;
    const int32_t tap = indices[n][q][j];
    float sum = 0.0f;
    if (tap >= 0 && tap < height * width) {
      sum = grad_confidence[n][0][q];
      for (index_t c = 0; c < channels; ++c)
        sum += static_cast<float>(grad_samples[n][c][q]) *
            static_cast<float>(features[n][c][tap / width][tap % width]);
    }
    grad_weights[n][q][j] = sum;
  }
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

void check_source_features(
    const torch::Tensor& f,
    const torch::Tensor& idx,
    const torch::Tensor& w) {
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
}

} // namespace

template <typename Evaluator>
std::vector<torch::Tensor> launch_source_stencil_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& opacity,
    int64_t height,
    int64_t width,
    int64_t tile_capacity,
    const Evaluator& evaluator,
    const torch::Tensor& uv0 = {},
    const torch::Tensor& uv1 = {}) {
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
  auto g = source_geometry(v, vi, opacity);
  if constexpr (Evaluator::shaded) {
    g.uv = uv0.data_ptr<float>();
    g.reference_uv = uv1.defined() ? uv1.data_ptr<float>() : nullptr;
  }
  const auto adjacency = source_build_adjacency(g, vi);
  const auto bins = build_tile_bins(
      v, vi, static_cast<int32_t>(height), static_cast<int32_t>(width), 1, tile_capacity);
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
    source_stencil_kernel<<<GET_BLOCKS(n * q, 128), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        p,
        evaluator,
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

template <typename Evaluator>
std::vector<torch::Tensor> launch_source_stencil_backward(
    const torch::TensorList& s,
    const torch::TensorList& grads,
    int64_t height,
    int64_t width,
    bool edge_gradients,
    const Evaluator& evaluator,
    const torch::Tensor& uv0 = {},
    const torch::Tensor& uv1 = {},
    const torch::Tensor& gu = {},
    const torch::Tensor& gu1 = {}) {
  TORCH_CHECK(s.size() == 16 && grads.size() == 3, "source features: invalid saved state");
  const auto& v = s[0];
  const c10::cuda::CUDAGuard guard(v.device());
  auto gv = torch::zeros_like(v), gb = torch::zeros_like(s[3]), go = torch::zeros_like(s[4]);
  const auto total = s[2].numel();
  if (total) {
    auto g = source_geometry(v, s[1], s[4]);
    if constexpr (Evaluator::shaded) {
      g.uv = uv0.data_ptr<float>();
      g.reference_uv = uv1.defined() ? uv1.data_ptr<float>() : nullptr;
    }
    const auto p = source_params(g, s[2], s[3], s[13], s[14], s[15], s[11], s[12], height, width);
    source_stencil_backward_kernel<<<
        GET_BLOCKS(total, 128),
        128,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        p,
        evaluator,
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
        go.data_ptr<float>(),
        gu.defined() ? gu.data_ptr<float>() : nullptr,
        gu1.defined() ? gu1.data_ptr<float>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {gv, gb, go};
}

std::vector<torch::Tensor> source_stencil_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& opacity,
    int64_t height,
    int64_t width,
    int64_t tile_capacity) {
  return launch_source_stencil_forward(
      v, vi, qf, qb, opacity, height, width, tile_capacity, SourceMaterialEvaluator{});
}

std::vector<torch::Tensor> source_stencil_backward(
    const torch::TensorList& saved,
    const torch::TensorList& grads,
    int64_t height,
    int64_t width,
    bool edges) {
  return launch_source_stencil_backward(
      saved, grads, height, width, edges, SourceMaterialEvaluator{});
}

namespace {

struct SourceTextureForward : TextureFragmentEvaluator {
  __device__ void
  evaluate(int32_t n, const TriangleData& t, const float* bary, float*, float& alpha) const {
    alpha = 1.0f;
    if (channels == 3) {
      return;
    }
    float u = 0.0f, v = 0.0f;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      u += bary[c] * t.u[c];
      v += bary[c] * t.v[c];
    }
    const int64_t offset = ((texture_batch == 1 ? 0 : n) * int64_t(channels) + 3) * height * width;
    float raw;
    sample_texture(
        texture + offset, 1, 1, height, width, 0, u, v, interpolation_mode, padding_mode, &raw);
    alpha = activate_texture_alpha(raw, alpha_activation);
  }
};

struct SourceTextureBackward {
  SourceTextureForward forward;
  float* texture_grad;

  __device__ void
  evaluate(int32_t n, const TriangleData& t, const float* bary, float* rgb, float& alpha) const {
    forward.evaluate(n, t, bary, rgb, alpha);
  }

  __device__ void backward(
      int32_t n,
      const TriangleData& t,
      const float* bary,
      const float*,
      float grad,
      float,
      FragmentAttributeGrads& attr) const {
    if (forward.channels == 3) {
      return;
    }
    float u = 0.0f, v = 0.0f;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      u += bary[c] * t.u[c];
      v += bary[c] * t.v[c];
    }
    const int64_t offset = ((forward.texture_batch == 1 ? 0 : n) * int64_t(forward.channels) + 3) *
        forward.height * forward.width;
    const float* texture = forward.texture + offset;
    float raw;
    sample_texture(
        texture,
        1,
        1,
        forward.height,
        forward.width,
        0,
        u,
        v,
        forward.interpolation_mode,
        forward.padding_mode,
        &raw);
    grad *= activate_texture_alpha_backward(raw, forward.alpha_activation);
    float gu = 0.0f, gv = 0.0f;
    sample_texture_backward(
        texture,
        texture_grad + offset,
        1,
        1,
        forward.height,
        forward.width,
        0,
        u,
        v,
        forward.interpolation_mode,
        forward.padding_mode,
        &grad,
        gu,
        gv);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      attr.u[c] += bary[c] * gu;
      attr.v[c] += bary[c] * gv;
      attr.bary[c] += gu * t.u[c] + gv * t.v[c];
    }
  }
};

void check_source_appearance(
    const torch::Tensor& v,
    const torch::TensorList& a,
    const SourceShadingOptions& o) {
  TORCH_CHECK(
      o.interpolation_mode >= 0 && o.interpolation_mode < 3 && o.padding_mode >= 0 &&
          o.padding_mode < 3 && o.alpha_activation >= 0 &&
          o.alpha_activation < kActivationModeCount,
      "source transmittance: invalid shading options");
  TORCH_CHECK(
      std::isfinite(o.negative_slope) && o.negative_slope >= 0.0f && o.negative_slope <= 1.0f,
      "source transmittance: negative_slope must be in [0, 1]");
  TORCH_CHECK(
      o.mlp ? (a.size() == 12 || a.size() == 14) : a.size() == 2,
      "source transmittance: invalid appearance tensor list");
  for (const auto& t : a) {
    source_check(t, at::kFloat, v.device());
  }
  for (int i = 0; i < (o.mlp ? 2 : 1); ++i) {
    TORCH_CHECK(
        a[i].dim() == 3 && a[i].size(0) == v.size(0) && a[i].size(1) == v.size(1) &&
            a[i].size(2) == 2,
        "source transmittance: UVs must have shape [N, V, 2]");
  }
  if (!o.mlp) {
    check_texture(a[1], v.size(0), "texture");
    TORCH_CHECK(
        a[1].size(1) == 3 || a[1].size(1) == 4,
        "source transmittance: texture must be RGB or RGBA");
    return;
  }
  check_texture(a[2], v.size(0), "texture0");
  check_texture(a[3], v.size(0), "texture1", true);
  int64_t inputs = a[2].size(1) + a[3].size(1);
  TORCH_CHECK(
      inputs > 0 && inputs <= kMaxFeatureChannels,
      "source transmittance: total feature channels must be in [1, 32]");
  const size_t layers = (a.size() - 4) / 2;
  for (size_t i = 0; i < layers; ++i) {
    const int64_t outputs = i + 1 == layers ? 4 : kMlpWidth;
    TORCH_CHECK(
        a[4 + i].dim() == 2 && a[4 + i].size(0) == outputs && a[4 + i].size(1) == inputs &&
            a[4 + layers + i].dim() == 1 && a[4 + layers + i].size(0) == outputs,
        "source transmittance: invalid MLP parameter shape");
    inputs = outputs;
  }
}

SourceTextureForward source_texture_evaluator(
    const torch::Tensor& texture,
    const SourceShadingOptions& o) {
  return {
      {texture.data_ptr<float>(),
       static_cast<int32_t>(texture.size(0)),
       static_cast<int32_t>(texture.size(1)),
       static_cast<int32_t>(texture.size(2)),
       static_cast<int32_t>(texture.size(3)),
       o.interpolation_mode,
       o.padding_mode,
       0,
       o.alpha_activation}};
}

template <int Layers>
MlpFragmentEvaluator<Layers, true> source_mlp_evaluator(
    const torch::TensorList& a,
    const SourceShadingOptions& o) {
  TinyMlpParams mlp{};
  mlp.input_channels = a[2].size(1) + a[3].size(1);
  mlp.negative_slope = o.negative_slope;
  mlp.alpha_activation = o.alpha_activation;
  for (int i = 0; i < Layers; ++i) {
    mlp.weights[i] = a[4 + i].data_ptr<float>();
    mlp.biases[i] = a[4 + Layers + i].data_ptr<float>();
  }
  return {
      a[2].data_ptr<float>(),
      a[3].data_ptr<float>(),
      static_cast<int32_t>(a[2].size(0)),
      static_cast<int32_t>(a[3].size(0)),
      static_cast<int32_t>(a[2].size(1)),
      static_cast<int32_t>(a[3].size(1)),
      static_cast<int32_t>(a[2].size(2)),
      static_cast<int32_t>(a[2].size(3)),
      static_cast<int32_t>(a[3].size(2)),
      static_cast<int32_t>(a[3].size(3)),
      o.interpolation_mode,
      o.padding_mode,
      mlp};
}

template <int Layers>
std::vector<torch::Tensor> source_mlp_backward(
    const torch::TensorList& s,
    const torch::TensorList& grads,
    const torch::TensorList& a,
    const torch::TensorList& ga,
    int64_t height,
    int64_t width,
    bool edges,
    const SourceShadingOptions& o) {
  TinyMlpGradParams mg{};
  for (int i = 0; i < Layers; ++i) {
    mg.weights[i] = ga[4 + i].data_ptr<float>();
    mg.biases[i] = ga[4 + Layers + i].data_ptr<float>();
  }
  SourceAppearanceEvaluator<MlpFragmentBackward<Layers, true>, true> evaluator{
      {source_mlp_evaluator<Layers>(a, o),
       ga[2].data_ptr<float>(),
       ga[3].data_ptr<float>(),
       mg,
       {}}};
  return launch_source_stencil_backward(
      s, grads, height, width, edges, evaluator, a[0], a[1], ga[0], ga[1]);
}

} // namespace

std::vector<torch::Tensor> source_stencil_appearance_forward(
    const torch::Tensor& v,
    const torch::Tensor& vi,
    const torch::Tensor& qf,
    const torch::Tensor& qb,
    const torch::Tensor& opacity,
    const torch::TensorList& a,
    int64_t height,
    int64_t width,
    const SourceShadingOptions& o,
    int64_t tile_capacity) {
  check_source_appearance(v, a, o);
  if (!o.mlp) {
    SourceAppearanceEvaluator<SourceTextureForward> evaluator{source_texture_evaluator(a[1], o)};
    return launch_source_stencil_forward(
        v, vi, qf, qb, opacity, height, width, tile_capacity, evaluator, a[0]);
  }
  if (a.size() == 12) {
    SourceAppearanceEvaluator<MlpFragmentEvaluator<4, true>> evaluator{
        source_mlp_evaluator<4>(a, o)};
    return launch_source_stencil_forward(
        v, vi, qf, qb, opacity, height, width, tile_capacity, evaluator, a[0], a[1]);
  }
  SourceAppearanceEvaluator<MlpFragmentEvaluator<5, true>> evaluator{source_mlp_evaluator<5>(a, o)};
  return launch_source_stencil_forward(
      v, vi, qf, qb, opacity, height, width, tile_capacity, evaluator, a[0], a[1]);
}

std::vector<torch::Tensor> source_stencil_appearance_backward(
    const torch::TensorList& s,
    const torch::TensorList& grads,
    const torch::TensorList& a,
    int64_t height,
    int64_t width,
    bool edges,
    const SourceShadingOptions& o) {
  const c10::cuda::CUDAGuard guard(s[0].device());
  std::vector<torch::Tensor> ga;
  for (const auto& t : a) {
    ga.push_back(torch::zeros_like(t));
  }
  std::vector<torch::Tensor> result;
  if (!o.mlp) {
    SourceAppearanceEvaluator<SourceTextureBackward> evaluator{
        {source_texture_evaluator(a[1], o), ga[1].data_ptr<float>()}};
    result =
        launch_source_stencil_backward(s, grads, height, width, edges, evaluator, a[0], {}, ga[0]);
  } else if (a.size() == 12) {
    result = source_mlp_backward<4>(s, grads, a, ga, height, width, edges, o);
  } else {
    result = source_mlp_backward<5>(s, grads, a, ga, height, width, edges, o);
  }
  result.insert(result.end(), ga.begin(), ga.end());
  return result;
}

namespace {
template <typename scalar_t, typename index_t>
void launch_source_sample_forward(
    int64_t count,
    const torch::Tensor& features,
    const torch::Tensor& indices,
    const torch::Tensor& weights,
    torch::Tensor& samples,
    torch::Tensor& confidence) {
  source_sample_kernel<scalar_t, index_t>
      <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<index_t>(count),
          getTensorInfoCompact<scalar_t, index_t, 4>(features),
          getTensorInfoCompact<int32_t, index_t, 3>(indices),
          getTensorInfoCompact<float, index_t, 3>(weights),
          getTensorInfoCompact<scalar_t, index_t, 3>(samples),
          getTensorInfoCompact<float, index_t, 3>(confidence));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename index_t>
void launch_source_sample_backward(
    const torch::Tensor& f,
    const torch::Tensor& idx,
    const torch::Tensor& weights,
    const torch::Tensor& gs,
    const torch::Tensor& gc,
    torch::Tensor& gf,
    torch::Tensor& gw) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (gs.numel()) {
    source_sample_image_backward_kernel<scalar_t, index_t>
        <<<GET_BLOCKS(gs.numel(), 256), 256, 0, stream>>>(
            static_cast<index_t>(gs.numel()),
            getTensorInfoCompact<int32_t, index_t, 3>(idx),
            getTensorInfoCompact<float, index_t, 3>(weights),
            getTensorInfoCompact<scalar_t, index_t, 3>(gs),
            getTensorInfoCompact<float, index_t, 4>(gf));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (idx.numel()) {
    source_sample_weight_backward_kernel<scalar_t, index_t>
        <<<GET_BLOCKS(idx.numel(), 256), 256, 0, stream>>>(
            static_cast<index_t>(idx.numel()),
            getTensorInfoCompact<scalar_t, index_t, 4>(f),
            getTensorInfoCompact<int32_t, index_t, 3>(idx),
            getTensorInfoCompact<scalar_t, index_t, 3>(gs),
            getTensorInfoCompact<float, index_t, 3>(gc),
            getTensorInfoCompact<float, index_t, 3>(gw));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
} // namespace

std::vector<torch::Tensor> source_sample_forward(
    const torch::Tensor& f,
    const torch::Tensor& idx,
    const torch::Tensor& weights) {
  check_source_features(f, idx, weights);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(f));
  auto out = torch::empty({f.size(0), f.size(1), idx.size(1)}, f.options());
  auto confidence = torch::empty({f.size(0), 1, idx.size(1)}, weights.options());
  const int64_t count = std::max(out.numel(), confidence.numel());
  if (count) {
    const bool index32 = at::native::canUse32BitIndexMath(f) &&
        at::native::canUse32BitIndexMath(idx) && at::native::canUse32BitIndexMath(weights) &&
        at::native::canUse32BitIndexMath(out) && at::native::canUse32BitIndexMath(confidence);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, f.scalar_type(), "source_sample_forward", [&] {
          if (index32)
            launch_source_sample_forward<scalar_t, int32_t>(
                count, f, idx, weights, out, confidence);
          else
            launch_source_sample_forward<scalar_t, int64_t>(
                count, f, idx, weights, out, confidence);
        });
  }
  return {out, confidence};
}

std::vector<torch::Tensor> source_sample_backward(
    const torch::Tensor& f,
    const torch::Tensor& idx,
    const torch::Tensor& weights,
    const torch::Tensor& gs,
    const torch::Tensor& gc) {
  check_source_features(f, idx, weights);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(f));
  auto gf = torch::zeros(f.sizes(), f.options().dtype(at::kFloat));
  auto gw = torch::empty_like(weights);
  bool index32 = true;
  for (const auto& t : {f, idx, weights, gs, gc, gf, gw})
    index32 = index32 && at::native::canUse32BitIndexMath(t);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, f.scalar_type(), "source_sample_backward", [&] {
        if (index32)
          launch_source_sample_backward<scalar_t, int32_t>(f, idx, weights, gs, gc, gf, gw);
        else
          launch_source_sample_backward<scalar_t, int64_t>(f, idx, weights, gs, gc, gf, gw);
      });
  return {gf.to(f.scalar_type()), gw};
}

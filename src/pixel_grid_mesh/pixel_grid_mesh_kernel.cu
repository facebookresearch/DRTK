// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_math_helper.h>
#include <kernel_utils.h>
#include <tensor_list.h>
#include <cub/cub.cuh>
#include <limits>

#include "pixel_grid_mesh_kernel.h"

using namespace math;

namespace {

template <typename scalar_t, typename index_t>
__device__ TVec3<scalar_t> read3(const scalar_t* p, index_t i) {
  return {p[3 * i], p[3 * i + 1], p[3 * i + 2]};
}

template <typename scalar_t, typename index_t>
__device__ void write3(scalar_t* p, index_t i, TVec3<scalar_t> v) {
  p[3 * i] = v.x;
  p[3 * i + 1] = v.y;
  p[3 * i + 2] = v.z;
}

template <typename scalar_t, typename index_t>
__device__ void add3(scalar_t* p, index_t i, TVec3<scalar_t> v) {
  atomicAdd(p + 3 * i, v.x);
  atomicAdd(p + 3 * i + 1, v.y);
  atomicAdd(p + 3 * i + 2, v.z);
}

template <typename scalar_t>
__device__ TVec3<scalar_t> mult(const TVec3<scalar_t> (&m)[3], TVec3<scalar_t> v) {
  return m[0] * v.x + m[1] * v.y + m[2] * v.z;
}

template <typename scalar_t, typename index_t>
__device__ scalar_t
cell_value(const TensorInfoCompact<scalar_t, index_t, 3>& value, index_t batch, index_t cell) {
  return value[batch][cell / value.sizes[2]][cell % value.sizes[2]];
}

template <typename scalar_t, typename index_t>
__device__ TVec3<scalar_t> cell_displacement(
    const TensorInfoCompact<scalar_t, index_t, 4>& value,
    index_t batch,
    index_t cell) {
  const auto xyz = value[batch][cell / value.sizes[2]][cell % value.sizes[2]];
  return {xyz[0], xyz[1], xyz[2]};
}

template <typename scalar_t, typename index_t>
__device__ void read_intrinsics(
    const TensorInfoCompact<scalar_t, index_t, 3>& intrinsics,
    index_t batch,
    TVec3<scalar_t> (&k)[3]) {
  for (int row = 0; row < 3; ++row) {
    const auto values = intrinsics[batch][row];
    k[row] = {values[0], values[1], values[2]};
  }
}

template <typename scalar_t>
__device__ void outer_add(scalar_t* g, TVec3<scalar_t> a, TVec3<scalar_t> b) {
  if (!g)
    return;
  const scalar_t av[3] = {a.x, a.y, a.z}, bv[3] = {b.x, b.y, b.z};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      atomicAdd(g + 3 * i + j, av[i] * bv[j]);
}

template <typename scalar_t>
__device__ TVec2<scalar_t> project(const TVec3<scalar_t> (&k)[3], TVec3<scalar_t> v) {
  const auto h = mul(k, v);
  return {h.x / h.z, h.y / h.z};
}

template <typename scalar_t>
__device__ TVec3<scalar_t> project_backward(
    const TVec3<scalar_t> (&k)[3],
    TVec3<scalar_t> v,
    TVec2<scalar_t> g,
    scalar_t* gk) {
  const auto h = mul(k, v);
  const TVec3<scalar_t> gh = {g.x / h.z, g.y / h.z, -(g.x * h.x + g.y * h.y) / (h.z * h.z)};
  outer_add(gk, gh, v);
  return mult<scalar_t>(k, gh);
}

template <typename scalar_t>
__device__ TVec2<scalar_t> unit(TVec2<scalar_t> v) {
  const scalar_t len = sqrt(dot(v, v));
  const scalar_t d = len > std::numeric_limits<scalar_t>::epsilon()
      ? len
      : std::numeric_limits<scalar_t>::epsilon();
  return v * (scalar_t(1) / d);
}

template <typename scalar_t>
__device__ TVec2<scalar_t> unit_backward(TVec2<scalar_t> v, TVec2<scalar_t> g) {
  const scalar_t len = sqrt(dot(v, v)), eps = std::numeric_limits<scalar_t>::epsilon();
  if (len <= eps)
    return g * (scalar_t(1) / eps);
  const auto t = v * (scalar_t(1) / len);
  return (g - t * dot(g, t)) * (scalar_t(1) / len);
}

template <typename index_t>
struct Grid {
  index_t h, w;
  at::TensorAccessor<bool, 2, at::RestrictPtrTraits, index_t> alive, right, down;
  __device__ bool live(index_t y, index_t x) const {
    return y >= 0 && y < h && x >= 0 && x < w && alive[y][x];
  }
  __device__ bool cut_r(index_t y, index_t x) const {
    return live(y, x) && live(y, x + 1) && !right[y][x];
  }
  __device__ bool cut_d(index_t y, index_t x) const {
    return live(y, x) && live(y + 1, x) && !down[y][x];
  }
  __device__ bool supported(index_t y, index_t x) const {
    index_t incident = index_t(live(y - 1, x - 1)) + index_t(live(y - 1, x)) +
        index_t(live(y, x - 1)) + index_t(live(y, x));
    index_t degree = index_t(cut_r(y - 1, x - 1)) + index_t(cut_r(y, x - 1)) +
        index_t(cut_d(y - 1, x - 1)) + index_t(cut_d(y - 1, x));
    return degree > 1 || (incident > 0 && incident < 4);
  }
  __device__ bool joined_r(index_t y, index_t x) const {
    return live(y, x) && live(y, x + 1) && right[y][x];
  }
  __device__ bool joined_d(index_t y, index_t x) const {
    return live(y, x) && live(y + 1, x) && down[y][x];
  }
};

template <typename index_t>
__device__ void lattice_corner(index_t raw, index_t w, index_t& y, index_t& x) {
  const index_t c = raw / 4, s = raw % 4;
  y = c / w + (s >= 2);
  x = c % w + (s == 1 || s == 2);
}

template <typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void realize_kernel(
    const index_t nthreads,
    TensorInfoCompact<bool, index_t, 3> alive_info,
    TensorInfoCompact<bool, index_t, 3> join_right_info,
    TensorInfoCompact<bool, index_t, 3> join_down_info,
    TensorInfoCompact<bool, index_t, 3> right_info,
    TensorInfoCompact<bool, index_t, 3> down_info) {
  const index_t h = alive_info.sizes[1], w = alive_info.sizes[2];
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / (h * w);
    const index_t c = index % (h * w);
    Grid<index_t> g{
        h, w, alive_info[batch_idx], join_right_info[batch_idx], join_down_info[batch_idx]};
    index_t y = c / g.w, x = c % g.w;
    if (x + 1 < g.w)
      right_info[batch_idx][y][x] = g.live(y, x) && g.live(y, x + 1) &&
          (!g.cut_r(y, x) || !(g.supported(y, x + 1) || g.supported(y + 1, x + 1)));
    if (y + 1 < g.h)
      down_info[batch_idx][y][x] = g.live(y, x) && g.live(y + 1, x) &&
          (!g.cut_d(y, x) || !(g.supported(y + 1, x) || g.supported(y + 1, x + 1)));
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void corner_kernel(
    const index_t nthreads,
    TensorInfoCompact<bool, index_t, 3> alive_info,
    TensorInfoCompact<bool, index_t, 3> join_right_info,
    TensorInfoCompact<bool, index_t, 3> join_down_info,
    TensorInfoCompact<scalar_t, index_t, 3> d_info,
    TensorInfoCompact<scalar_t, index_t, 4> displacement_info,
    TensorInfoCompact<scalar_t, index_t, 3> k_info,
    TensorInfoCompact<index_t, index_t, 2> owners_info,
    TensorInfoCompact<scalar_t, index_t, 3> core_info,
    TensorInfoCompact<uint8_t, index_t, 2> sizes_info) {
  const index_t h = alive_info.sizes[1], w = alive_info.sizes[2];
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / (4 * h * w);
    const index_t raw = index % (4 * h * w);
    Grid<index_t> g{
        h, w, alive_info[batch_idx], join_right_info[batch_idx], join_down_info[batch_idx]};
    index_t* owners = owners_info.data + batch_idx * owners_info.strides[0];
    scalar_t* core = core_info.data + batch_idx * core_info.strides[0];
    uint8_t* sizes = sizes_info.data + batch_idx * sizes_info.strides[0];
    if (!g.alive[raw / (4 * g.w)][(raw / 4) % g.w]) {
      owners[raw] = -1;
      sizes[raw] = 0;
      continue;
    }
    index_t y, x;
    lattice_corner(raw, g.w, y, x);
    const index_t cy[4] = {y - 1, y - 1, y, y}, cx[4] = {x - 1, x, x, x - 1},
                  local[4] = {2, 3, 0, 1};
    index_t ids[4], label[4], me = 0;
    for (index_t i = 0; i < 4; ++i) {
      ids[i] = g.live(cy[i], cx[i]) ? 4 * (cy[i] * g.w + cx[i]) + local[i] : -1;
      label[i] = i;
      if (ids[i] == raw)
        me = i;
    }
    bool link[4] = {
        g.joined_r(y - 1, x - 1),
        g.joined_d(y - 1, x),
        g.joined_r(y, x - 1),
        g.joined_d(y - 1, x - 1)};
    for (index_t step = 0; step < 3; ++step) {
      index_t next[4];
      for (index_t i = 0; i < 4; ++i) {
        next[i] = label[i];
        if (link[i])
          next[i] = min(next[i], label[(i + 1) % 4]);
        if (link[(i + 3) % 4])
          next[i] = min(next[i], label[(i + 3) % 4]);
      }
      for (index_t i = 0; i < 4; ++i)
        label[i] = next[i];
    }
    index_t owner = raw, count = 0;
    for (index_t i = 0; i < 4; ++i)
      if (ids[i] >= 0 && label[i] == label[me]) {
        owner = min(owner, ids[i]);
        ++count;
      }
    owners[raw] = owner;
    sizes[raw] = count;
    if (owner != raw)
      continue;
    TVec3<scalar_t> k[3];
    read_intrinsics(k_info, batch_idx, k);
    TVec3<scalar_t> inv[3];
    inverse(k, inv);
    const auto ray = mul(
        inv, TVec3<scalar_t>{scalar_t(x) - scalar_t(.5), scalar_t(y) - scalar_t(.5), scalar_t(1)});
    TVec3<scalar_t> v = {0, 0, 0};
    for (index_t i = 0; i < 4; ++i)
      if (ids[i] >= 0 && label[i] == label[me]) {
        index_t c = ids[i] / 4;
        v = v + ray * cell_value(d_info, batch_idx, c) +
            cell_displacement(displacement_info, batch_idx, c);
      }
    write3(core, raw, v * (scalar_t(1) / scalar_t(count)));
  }
}

template <typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void boundary_kernel(
    const index_t nthreads,
    TensorInfoCompact<bool, index_t, 3> alive_info,
    TensorInfoCompact<bool, index_t, 3> join_right_info,
    TensorInfoCompact<bool, index_t, 3> join_down_info,
    TensorInfoCompact<index_t, index_t, 2> owners_info,
    TensorInfoCompact<index_t, index_t, 2> outgoing_info,
    TensorInfoCompact<index_t, index_t, 2> incoming_info,
    TensorInfoCompact<uint8_t, index_t, 2> modes_info) {
  const index_t h = alive_info.sizes[1], w = alive_info.sizes[2];
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / (4 * h * w);
    const index_t e = index % (4 * h * w);
    Grid<index_t> g{
        h, w, alive_info[batch_idx], join_right_info[batch_idx], join_down_info[batch_idx]};
    const index_t* owners = owners_info.data + batch_idx * owners_info.strides[0];
    index_t* outgoing = outgoing_info.data + batch_idx * outgoing_info.strides[0];
    index_t* incoming = incoming_info.data + batch_idx * incoming_info.strides[0];
    uint8_t* modes = modes_info.data + batch_idx * modes_info.strides[0];
    index_t c = e / 4, s = e % 4, y = c / g.w, x = c % g.w;
    bool joined = s == 0
        ? g.joined_d(y - 1, x)
        : (s == 1 ? g.joined_r(y, x) : (s == 2 ? g.joined_d(y, x) : g.joined_r(y, x - 1)));
    bool boundary = g.alive[c / g.w][c % g.w] && !joined;
    modes[e] = boundary ? 1 : 0;
    if (boundary) {
      outgoing[owners[e]] = e;
      incoming[owners[4 * c + (s + 1) % 4]] = e;
    }
  }
}

template <typename scalar_t, typename index_t>
struct EdgeGeometry {
  index_t a, b, p, previous, following;
  TVec2<scalar_t> tangent, previous_tangent, normal, previous_normal, miter;
  scalar_t width, denominator;
};

template <typename scalar_t, typename index_t>
__device__ EdgeGeometry<scalar_t, index_t> edge_geometry(
    index_t e,
    const index_t* owners,
    const scalar_t* core,
    const index_t* outgoing,
    const index_t* incoming,
    const TVec3<scalar_t> (&k)[3],
    const TensorInfoCompact<scalar_t, index_t, 3>& width,
    index_t batch) {
  EdgeGeometry<scalar_t, index_t> r;
  r.a = owners[e];
  r.b = owners[4 * (e / 4) + (e % 4 + 1) % 4];
  r.previous = incoming[r.a];
  r.following = outgoing[r.b];
  r.p = owners[r.previous];
  const auto a = project<scalar_t>(k, read3(core, r.a)), b = project<scalar_t>(k, read3(core, r.b)),
             p = project<scalar_t>(k, read3(core, r.p));
  r.tangent = unit<scalar_t>(b - a);
  r.previous_tangent = unit<scalar_t>(a - p);
  r.normal = {r.tangent.y, -r.tangent.x};
  r.previous_normal = {r.previous_tangent.y, -r.previous_tangent.x};
  r.denominator = scalar_t(1) + dot(r.normal, r.previous_normal);
  scalar_t denom = r.denominator > std::numeric_limits<scalar_t>::epsilon()
      ? r.denominator
      : std::numeric_limits<scalar_t>::epsilon();
  r.miter = (r.normal + r.previous_normal) * (scalar_t(1) / denom);
  r.width =
      scalar_t(.5) * (cell_value(width, batch, e / 4) + cell_value(width, batch, r.previous / 4));
  return r;
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void mode_kernel(
    const index_t nthreads,
    index_t edges,
    TensorInfoCompact<index_t, index_t, 2> owners_info,
    TensorInfoCompact<scalar_t, index_t, 3> core_info,
    TensorInfoCompact<index_t, index_t, 2> outgoing_info,
    TensorInfoCompact<index_t, index_t, 2> incoming_info,
    TensorInfoCompact<scalar_t, index_t, 3> k_info,
    TensorInfoCompact<scalar_t, index_t, 3> width_info,
    TensorInfoCompact<uint8_t, index_t, 2> modes_info) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / (edges);
    const index_t e = index % (edges);
    const index_t* owners = owners_info.data + batch_idx * owners_info.strides[0];
    const scalar_t* core = core_info.data + batch_idx * core_info.strides[0];
    const index_t* outgoing = outgoing_info.data + batch_idx * outgoing_info.strides[0];
    const index_t* incoming = incoming_info.data + batch_idx * incoming_info.strides[0];
    uint8_t* modes = modes_info.data + batch_idx * modes_info.strides[0];
    if (modes[e] == 0)
      continue;
    TVec3<scalar_t> k[3];
    read_intrinsics(k_info, batch_idx, k);
    const auto r = edge_geometry(e, owners, core, outgoing, incoming, k, width_info, batch_idx);
    modes[e] = r.denominator < scalar_t(1e-4) ? 3 : (dot(r.miter, r.miter) > scalar_t(16) ? 2 : 1);
  }
}

template <typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void count_kernel(
    const index_t nthreads,
    index_t n,
    TensorInfoCompact<bool, index_t, 3> alive_info,
    TensorInfoCompact<index_t, index_t, 2> owners_info,
    TensorInfoCompact<index_t, index_t, 2> outgoing_info,
    TensorInfoCompact<uint8_t, index_t, 2> modes_info,
    bool feather,
    TensorInfoCompact<int32_t, index_t, 2> vmask_info,
    TensorInfoCompact<int32_t, index_t, 2> fmask_info,
    TensorInfoCompact<index_t, index_t, 2> vcount_info,
    TensorInfoCompact<index_t, index_t, 2> fcount_info) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / (n + 1);
    const index_t c = index % (n + 1);
    const index_t* owners = owners_info.data + batch_idx * owners_info.strides[0];
    const index_t* outgoing = outgoing_info.data + batch_idx * outgoing_info.strides[0];
    const uint8_t* modes = modes_info.data + batch_idx * modes_info.strides[0];
    int32_t* vmask = vmask_info.data + batch_idx * vmask_info.strides[0];
    int32_t* fmask = fmask_info.data + batch_idx * fmask_info.strides[0];
    index_t* vcount = vcount_info.data + batch_idx * vcount_info.strides[0];
    index_t* fcount = fcount_info.data + batch_idx * fcount_info.strides[0];
    if (c == n) {
      vcount[c] = fcount[c] = 0;
      continue;
    }
    unsigned v = 0, f = cell_value(alive_info, batch_idx, c) ? 3 : 0;
    for (index_t s = 0; s < 4; ++s) {
      index_t e = 4 * c + s;
      if (owners[e] == e)
        v |= 1u << s;
      if (feather && modes[e]) {
        v |= 1u << (4 + s);
        f |= 3u << (2 + 4 * s);
        index_t next = outgoing[owners[4 * c + (s + 1) % 4]];
        if (modes[next] >= 2)
          v |= 1u << (8 + s);
        if (modes[e] == 2)
          f |= 1u << (4 + 4 * s);
        if (modes[e] == 3) {
          v |= 1u << (12 + s);
          f |= 3u << (4 + 4 * s);
        }
      }
    }
    vmask[c] = v;
    fmask[c] = f;
    vcount[c] = __popc(v);
    fcount[c] = __popc(f);
  }
}

template <typename index_t>
__device__ index_t
compact_vertex(index_t cell, index_t slot, const int32_t* mask, const index_t* prefix) {
  return prefix[cell] + __popc(static_cast<unsigned>(mask[cell]) & ((1u << slot) - 1));
}

template <typename index_t>
__device__ index_t core_vertex(index_t raw, const int32_t* mask, const index_t* prefix) {
  return compact_vertex(raw / 4, raw % 4, mask, prefix);
}

template <typename index_t>
__device__ index_t outer_first(index_t e, const int32_t* mask, const index_t* prefix) {
  return compact_vertex(e / 4, 4 + e % 4, mask, prefix);
}

template <typename index_t>
__device__ index_t outer_last(
    index_t e,
    const index_t* owners,
    const index_t* outgoing,
    const uint8_t* modes,
    const int32_t* mask,
    const index_t* prefix) {
  index_t next = outgoing[owners[4 * (e / 4) + (e % 4 + 1) % 4]];
  return modes[next] >= 2 ? compact_vertex(e / 4, 8 + e % 4, mask, prefix)
                          : outer_first(next, mask, prefix);
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void vertex_kernel(
    const index_t nthreads,
    index_t n,
    index_t h,
    index_t w,
    TensorInfoCompact<index_t, index_t, 2> owners_info,
    TensorInfoCompact<scalar_t, index_t, 3> core_info,
    TensorInfoCompact<index_t, index_t, 2> outgoing_info,
    TensorInfoCompact<index_t, index_t, 2> incoming_info,
    TensorInfoCompact<uint8_t, index_t, 2> modes_info,
    TensorInfoCompact<int32_t, index_t, 2> mask_info,
    TensorInfoCompact<index_t, index_t, 2> prefix_info,
    TensorInfoCompact<scalar_t, index_t, 3> k_info,
    TensorInfoCompact<scalar_t, index_t, 3> width_info,
    TensorInfoCompact<scalar_t, index_t, 3> v_info,
    TensorInfoCompact<scalar_t, index_t, 3> uv0_info,
    TensorInfoCompact<scalar_t, index_t, 3> uv1_info,
    TensorInfoCompact<scalar_t, index_t, 2> opacity_info,
    index_t vertex_stride) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / n;
    const index_t c = index % n;
    const index_t* owners = owners_info.data + batch_idx * owners_info.strides[0];
    const scalar_t* core = core_info.data + batch_idx * core_info.strides[0];
    const index_t* outgoing = outgoing_info.data + batch_idx * outgoing_info.strides[0];
    const index_t* incoming = incoming_info.data + batch_idx * incoming_info.strides[0];
    const uint8_t* modes = modes_info.data + batch_idx * modes_info.strides[0];
    const int32_t* mask = mask_info.data + batch_idx * mask_info.strides[0];
    const index_t* prefix = prefix_info.data + batch_idx * prefix_info.strides[0];
    scalar_t* v = v_info.data + batch_idx * v_info.strides[0];
    scalar_t* uv0 = uv0_info.data + batch_idx * uv0_info.strides[0];
    scalar_t* uv1 = uv1_info.data + batch_idx * uv1_info.strides[0];
    scalar_t* opacity =
        opacity_info.data ? opacity_info.data + batch_idx * opacity_info.strides[0] : nullptr;
    for (index_t raw = prefix[n] + c; raw < vertex_stride; raw += n) {
      write3(v, raw, TVec3<scalar_t>{0, 0, 1});
      uv0[2 * raw] = uv0[2 * raw + 1] = 0;
      uv1[2 * raw] = uv1[2 * raw + 1] = 0;
      if (opacity)
        opacity[raw] = 0;
    }
    unsigned remaining = static_cast<unsigned>(mask[c]);
    while (remaining) {
      const index_t s = __ffs(remaining) - 1;
      remaining &= remaining - 1;
      TVec3<scalar_t> k[3];
      read_intrinsics(k_info, batch_idx, k);
      index_t vertex = compact_vertex(c, s, mask, prefix), anchor = 4 * c + s;
      TVec3<scalar_t> value;
      if (s < 4)
        value = read3(core, anchor);
      else {
        index_t e = 4 * c + s % 4;
        auto r = edge_geometry(e, owners, core, outgoing, incoming, k, width_info, batch_idx);
        TVec2<scalar_t> offset;
        anchor = r.a;
        if (s < 8)
          offset = (modes[e] >= 2 ? r.normal : r.miter) * r.width;
        else if (s < 12) {
          anchor = r.b;
          scalar_t wid = scalar_t(.5) *
              (cell_value(width_info, batch_idx, c) +
               cell_value(width_info, batch_idx, r.following / 4));
          offset = r.normal * wid;
        } else
          offset = r.previous_tangent * r.width;
        TVec3<scalar_t> inv[3];
        inverse(k, inv);
        value = read3(core, anchor);
        value = value + mul(inv, TVec3<scalar_t>{offset.x, offset.y, scalar_t(0)}) * value.z;
      }
      write3(v, vertex, value);
      index_t y, x;
      lattice_corner(anchor, w, y, x);
      uv0[2 * vertex] = scalar_t(x) / scalar_t(w);
      uv0[2 * vertex + 1] = scalar_t(y) / scalar_t(h);
      const auto pixel = project<scalar_t>(k, value);
      uv1[2 * vertex] = (pixel.x + scalar_t(.5)) / scalar_t(w);
      uv1[2 * vertex + 1] = (pixel.y + scalar_t(.5)) / scalar_t(h);
      if (opacity)
        opacity[vertex] = s < 4 ? scalar_t(1) : scalar_t(0);
    }
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void face_kernel(
    const index_t nthreads,
    index_t n,
    index_t w,
    TensorInfoCompact<index_t, index_t, 2> owners_info,
    TensorInfoCompact<scalar_t, index_t, 3> core_info,
    TensorInfoCompact<scalar_t, index_t, 3> depth_info,
    TensorInfoCompact<scalar_t, index_t, 4> displacement_info,
    TensorInfoCompact<scalar_t, index_t, 3> k_info,
    TensorInfoCompact<bool, index_t, 3> diagonal_info,
    TensorInfoCompact<index_t, index_t, 2> outgoing_info,
    TensorInfoCompact<index_t, index_t, 2> incoming_info,
    TensorInfoCompact<uint8_t, index_t, 2> modes_info,
    TensorInfoCompact<int32_t, index_t, 2> vmask_info,
    TensorInfoCompact<index_t, index_t, 2> vp_info,
    TensorInfoCompact<int32_t, index_t, 2> fmask_info,
    TensorInfoCompact<index_t, index_t, 2> fp_info,
    TensorInfoCompact<int32_t, index_t, 3> faces_info,
    TensorInfoCompact<int64_t, index_t, 2> cells_info,
    TensorInfoCompact<int64_t, index_t, 2> ids_info,
    index_t face_stride) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / n;
    const index_t c = index % n;
    const index_t* owners = owners_info.data + batch_idx * owners_info.strides[0];
    const scalar_t* core = core_info.data + batch_idx * core_info.strides[0];
    const index_t* outgoing = outgoing_info.data + batch_idx * outgoing_info.strides[0];
    const index_t* incoming = incoming_info.data + batch_idx * incoming_info.strides[0];
    const uint8_t* modes = modes_info.data + batch_idx * modes_info.strides[0];
    const int32_t* vmask = vmask_info.data + batch_idx * vmask_info.strides[0];
    const index_t* vp = vp_info.data + batch_idx * vp_info.strides[0];
    const int32_t* fmask = fmask_info.data + batch_idx * fmask_info.strides[0];
    const index_t* fp = fp_info.data + batch_idx * fp_info.strides[0];
    int32_t* faces = faces_info.data + batch_idx * faces_info.strides[0];
    int64_t* cells =
        cells_info.data ? cells_info.data + batch_idx * cells_info.strides[0] : nullptr;
    int64_t* ids = ids_info.data ? ids_info.data + batch_idx * ids_info.strides[0] : nullptr;
    for (index_t raw = fp[n] + c; raw < face_stride; raw += n) {
      faces[3 * raw] = faces[3 * raw + 1] = faces[3 * raw + 2] = -1;
      if (cells)
        cells[raw] = -1;
      if (ids)
        ids[raw] = -1;
    }
    unsigned remaining = static_cast<unsigned>(fmask[c]);
    while (remaining) {
      const index_t s = __ffs(remaining) - 1;
      remaining &= remaining - 1;
      index_t f = fp[c] + __popc(static_cast<unsigned>(fmask[c]) & ((1u << s) - 1));
      index_t a, b, d, id;
      if (s < 2) {
        index_t q[4];
        for (index_t i = 0; i < 4; ++i)
          q[i] = owners[4 * c + i];
        bool chosen;
        if (diagonal_info.data)
          chosen = cell_value(diagonal_info, batch_idx, c);
        else {
          TVec3<scalar_t> k[3];
          read_intrinsics(k_info, batch_idx, k);
          TVec3<scalar_t> inv[3];
          inverse(k, inv);
          const auto target =
              mul(inv, TVec3<scalar_t>{scalar_t(c % w), scalar_t(c / w), scalar_t(1)}) *
                  cell_value(depth_info, batch_idx, c) +
              cell_displacement(displacement_info, batch_idx, c);
          auto d02 = (read3(core, q[0]) + read3(core, q[2])) * scalar_t(.5) + target * scalar_t(-1);
          auto d13 = (read3(core, q[1]) + read3(core, q[3])) * scalar_t(.5) + target * scalar_t(-1);
          chosen = dot(d13, d13) < dot(d02, d02);
        }
        const index_t pattern0[6] = {0, 1, 2, 0, 2, 3}, pattern1[6] = {0, 1, 3, 1, 2, 3};
        const index_t* pat = chosen ? pattern1 : pattern0;
        a = core_vertex(q[pat[3 * s]], vmask, vp);
        b = core_vertex(q[pat[3 * s + 1]], vmask, vp);
        d = core_vertex(q[pat[3 * s + 2]], vmask, vp);
        id = 32 * c + s;
      } else {
        index_t side = (s - 2) / 4, part = (s - 2) % 4, e = 4 * c + side;
        index_t start = owners[e], end = owners[4 * c + (side + 1) % 4],
                inner = core_vertex(start, vmask, vp), first = outer_first(e, vmask, vp);
        id = 32 * c + 2 + 6 * side + part;
        if (part == 0) {
          a = inner;
          b = outer_last(e, owners, outgoing, modes, vmask, vp);
          d = core_vertex(end, vmask, vp);
        } else if (part == 1) {
          a = inner;
          b = first;
          d = outer_last(e, owners, outgoing, modes, vmask, vp);
        } else {
          index_t prev = incoming[start];
          a = inner;
          if (modes[e] == 2) {
            b = outer_last(prev, owners, outgoing, modes, vmask, vp);
            d = first;
          } else {
            index_t cap = compact_vertex(c, 12 + side, vmask, vp);
            ++id;
            if (part == 2) {
              b = outer_last(prev, owners, outgoing, modes, vmask, vp);
              d = cap;
            } else {
              b = cap;
              d = first;
            }
          }
        }
      }
      faces[3 * f] = a;
      faces[3 * f + 1] = b;
      faces[3 * f + 2] = d;
      if (cells)
        cells[f] = c;
      if (ids)
        ids[f] = id;
    }
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void projection_backward_kernel(
    const index_t nthreads,
    index_t vertices,
    index_t h,
    index_t w,
    TensorInfoCompact<scalar_t, index_t, 3> v_info,
    TensorInfoCompact<scalar_t, index_t, 3> k_info,
    TensorInfoCompact<scalar_t, index_t, 3> gv_info,
    TensorInfoCompact<scalar_t, index_t, 3> guv_info,
    TensorInfoCompact<scalar_t, index_t, 3> result_info,
    TensorInfoCompact<scalar_t, index_t, 3> gk_info,
    TensorInfoCompact<index_t, index_t, 2> vertex_prefix_info) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / (vertices);
    const index_t i = index % (vertices);
    const scalar_t* v = v_info.data + batch_idx * v_info.strides[0];
    scalar_t* result = result_info.data + batch_idx * result_info.strides[0];
    scalar_t* gk = gk_info.data ? gk_info.data + batch_idx * gk_info.strides[0] : nullptr;
    const index_t* vertex_prefix =
        vertex_prefix_info.data + batch_idx * vertex_prefix_info.strides[0];
    if (i >= vertex_prefix[h * w])
      continue;
    TVec3<scalar_t> k[3];
    read_intrinsics(k_info, batch_idx, k);
    TVec3<scalar_t> g = gv_info.data
        ? TVec3<
              scalar_t>{gv_info[batch_idx][i][0], gv_info[batch_idx][i][1], gv_info[batch_idx][i][2]}
        : TVec3<scalar_t>{0, 0, 0};
    if (guv_info.data)
      g = g +
          project_backward<scalar_t>(
              k,
              read3(v, i),
              TVec2<scalar_t>{
                  guv_info[batch_idx][i][0] / scalar_t(w), guv_info[batch_idx][i][1] / scalar_t(h)},
              gk);
    write3(result, i, g);
  }
}

template <typename scalar_t, typename index_t>
__device__ void tangent_backward(
    index_t a,
    index_t b,
    TVec2<scalar_t> gradient,
    const scalar_t* core,
    const TVec3<scalar_t> (&k)[3],
    scalar_t* gcore,
    scalar_t* gk) {
  const auto pa = project<scalar_t>(k, read3(core, a)), pb = project<scalar_t>(k, read3(core, b));
  auto edge_grad = unit_backward<scalar_t>(pb - pa, gradient);
  add3(gcore, a, project_backward<scalar_t>(k, read3(core, a), edge_grad * scalar_t(-1), gk));
  add3(gcore, b, project_backward<scalar_t>(k, read3(core, b), edge_grad, gk));
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void feather_backward_kernel(
    const index_t nthreads,
    index_t n,
    TensorInfoCompact<index_t, index_t, 2> owners_info,
    TensorInfoCompact<scalar_t, index_t, 3> core_info,
    TensorInfoCompact<index_t, index_t, 2> outgoing_info,
    TensorInfoCompact<index_t, index_t, 2> incoming_info,
    TensorInfoCompact<uint8_t, index_t, 2> modes_info,
    TensorInfoCompact<int32_t, index_t, 2> mask_info,
    TensorInfoCompact<index_t, index_t, 2> prefix_info,
    TensorInfoCompact<scalar_t, index_t, 3> k_info,
    TensorInfoCompact<scalar_t, index_t, 3> width_info,
    TensorInfoCompact<scalar_t, index_t, 3> grad_info,
    TensorInfoCompact<scalar_t, index_t, 3> gcore_info,
    TensorInfoCompact<scalar_t, index_t, 3> gwidth_info,
    TensorInfoCompact<scalar_t, index_t, 3> gk_info) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / (16 * n);
    const index_t raw = index % (16 * n);
    const index_t* owners = owners_info.data + batch_idx * owners_info.strides[0];
    const scalar_t* core = core_info.data + batch_idx * core_info.strides[0];
    const index_t* outgoing = outgoing_info.data + batch_idx * outgoing_info.strides[0];
    const index_t* incoming = incoming_info.data + batch_idx * incoming_info.strides[0];
    const uint8_t* modes = modes_info.data + batch_idx * modes_info.strides[0];
    const int32_t* mask = mask_info.data + batch_idx * mask_info.strides[0];
    const index_t* prefix = prefix_info.data + batch_idx * prefix_info.strides[0];
    const scalar_t* grad = grad_info.data + batch_idx * grad_info.strides[0];
    scalar_t* gcore =
        gcore_info.data ? gcore_info.data + batch_idx * gcore_info.strides[0] : nullptr;
    scalar_t* gwidth =
        gwidth_info.data ? gwidth_info.data + batch_idx * gwidth_info.strides[0] : nullptr;
    scalar_t* gk = gk_info.data ? gk_info.data + batch_idx * gk_info.strides[0] : nullptr;
    index_t c = raw / 16, s = raw % 16;
    if (s < 4 || !(static_cast<unsigned>(mask[c]) & (1u << s)))
      continue;
    TVec3<scalar_t> k[3];
    read_intrinsics(k_info, batch_idx, k);
    index_t e = 4 * c + s % 4;
    auto r = edge_geometry(e, owners, core, outgoing, incoming, k, width_info, batch_idx);
    index_t anchor = r.a, other = r.previous / 4;
    scalar_t wid = r.width;
    TVec2<scalar_t> direction;
    if (s < 8)
      direction = modes[e] >= 2 ? r.normal : r.miter;
    else if (s < 12) {
      anchor = r.b;
      other = r.following / 4;
      wid = scalar_t(.5) *
          (cell_value(width_info, batch_idx, c) + cell_value(width_info, batch_idx, other));
      direction = r.normal;
    } else
      direction = r.previous_tangent;
    const auto g = read3(grad, compact_vertex(c, s, mask, prefix));
    const auto pos = read3(core, anchor);
    TVec3<scalar_t> inv[3];
    inverse(k, inv);
    const auto ray = mul(inv, TVec3<scalar_t>{direction.x * wid, direction.y * wid, scalar_t(0)});
    add3(gcore, anchor, TVec3<scalar_t>{g.x, g.y, g.z + dot(g, ray)});
    const auto gdelta = mult<scalar_t>(inv, g) * pos.z;
    outer_add(gk, gdelta * scalar_t(-1), ray);
    const TVec2<scalar_t> gd = {gdelta.x, gdelta.y};
    const scalar_t gw = dot(gd, direction) * scalar_t(.5);
    if (gwidth) {
      atomicAdd(gwidth + c, gw);
      atomicAdd(gwidth + other, gw);
    }
    const auto gdir = gd * wid;
    if (s >= 12) {
      tangent_backward(r.p, r.a, gdir, core, k, gcore, gk);
      continue;
    }
    TVec2<scalar_t> gn = gdir, gp = {0, 0};
    if (s < 8 && modes[e] == 1) {
      const scalar_t denom = r.denominator,
                     coeff = dot(gdir, r.normal + r.previous_normal) / (denom * denom);
      gn = gdir * (scalar_t(1) / denom) - r.previous_normal * coeff;
      gp = gdir * (scalar_t(1) / denom) - r.normal * coeff;
    }
    tangent_backward(r.a, r.b, TVec2<scalar_t>{-gn.y, gn.x}, core, k, gcore, gk);
    if (s < 8 && modes[e] == 1)
      tangent_backward(r.p, r.a, TVec2<scalar_t>{-gp.y, gp.x}, core, k, gcore, gk);
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void corner_backward_kernel(
    const index_t nthreads,
    index_t n,
    index_t w,
    TensorInfoCompact<scalar_t, index_t, 3> depth_info,
    TensorInfoCompact<scalar_t, index_t, 3> k_info,
    TensorInfoCompact<index_t, index_t, 2> owners_info,
    TensorInfoCompact<uint8_t, index_t, 2> sizes_info,
    TensorInfoCompact<int32_t, index_t, 2> mask_info,
    TensorInfoCompact<index_t, index_t, 2> prefix_info,
    TensorInfoCompact<scalar_t, index_t, 3> grad_info,
    TensorInfoCompact<scalar_t, index_t, 3> gcore_info,
    TensorInfoCompact<scalar_t, index_t, 3> gd_info,
    TensorInfoCompact<scalar_t, index_t, 4> gdelta_info,
    TensorInfoCompact<scalar_t, index_t, 3> gk_info) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t batch_idx = index / (n);
    const index_t c = index % (n);
    const index_t* owners = owners_info.data + batch_idx * owners_info.strides[0];
    const uint8_t* sizes = sizes_info.data + batch_idx * sizes_info.strides[0];
    const int32_t* mask = mask_info.data + batch_idx * mask_info.strides[0];
    const index_t* prefix = prefix_info.data + batch_idx * prefix_info.strides[0];
    const scalar_t* grad = grad_info.data + batch_idx * grad_info.strides[0];
    const scalar_t* gcore =
        gcore_info.data ? gcore_info.data + batch_idx * gcore_info.strides[0] : nullptr;
    scalar_t* gd = gd_info.data + batch_idx * gd_info.strides[0];
    scalar_t* gdelta = gdelta_info.data + batch_idx * gdelta_info.strides[0];
    scalar_t* gk = gk_info.data ? gk_info.data + batch_idx * gk_info.strides[0] : nullptr;
    TVec3<scalar_t> k[3];
    read_intrinsics(k_info, batch_idx, k);
    scalar_t d = 0;
    TVec3<scalar_t> delta = {0, 0, 0};
    TVec3<scalar_t> inv[3];
    inverse(k, inv);
    for (index_t s = 0; s < 4; ++s) {
      index_t raw = 4 * c + s, owner = owners[raw];
      if (owner < 0)
        continue;
      auto g = read3(grad, core_vertex(owner, mask, prefix));
      if (gcore)
        g = g + read3(gcore, owner);
      g = g * (scalar_t(1) / scalar_t(sizes[raw]));
      index_t y, x;
      lattice_corner(raw, w, y, x);
      const auto ray =
          mul(inv,
              TVec3<scalar_t>{scalar_t(x) - scalar_t(.5), scalar_t(y) - scalar_t(.5), scalar_t(1)});
      d += dot(g, ray);
      delta = delta + g;
      if (gk)
        outer_add(
            gk,
            mult<scalar_t>(inv, g) * (scalar_t(-1) * cell_value(depth_info, batch_idx, c)),
            ray);
    }
    gd[c] = d;
    write3(gdelta, c, delta);
  }
}

template <typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void local_prefix_kernel(
    const index_t nthreads,
    TensorInfoCompact<index_t, index_t, 3> cumulative,
    TensorInfoCompact<index_t, index_t, 3> prefix,
    int64_t vertex_capacity,
    int64_t face_capacity) {
  const index_t row_size = prefix.sizes[2];
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t count = cumulative.data[index] - cumulative.data[(index / row_size) * row_size];
    if (vertex_capacity >= 0 && index % row_size == row_size - 1) {
      const int64_t capacity = index < prefix.strides[0] ? vertex_capacity : face_capacity;
      CUDA_KERNEL_ASSERT_MSG(count <= capacity, "pixel_grid_mesh(): output_capacity is too small");
    }
    prefix.data[index] = count;
  }
}

template <typename index_t>
std::vector<torch::Tensor> pixel_grid_mesh_forward_impl(
    const torch::Tensor& depth,
    const torch::Tensor& displacement,
    const torch::Tensor& intrinsics,
    const torch::Tensor& alive,
    const torch::Tensor& right,
    const torch::Tensor& down,
    const torch::Tensor& diagonal,
    const torch::Tensor& width,
    bool metadata,
    int64_t vertex_capacity,
    int64_t face_capacity) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t batch = depth.size(0), h = depth.size(1), w = depth.size(2), n = h * w;
  const bool feather = width.dim() == 3;
  const auto indices = depth.options().dtype(c10::CppTypeToScalarType<index_t>::value);
  const auto ints = depth.options().dtype(at::kInt), bytes = depth.options().dtype(at::kByte);
  auto er = torch::empty(right.sizes(), right.options()),
       ed = torch::empty(down.sizes(), down.options());
  auto owners = torch::empty({batch, 4 * n}, indices);
  auto core = torch::zeros({batch, 4 * n, 3}, depth.options());
  auto sizes = torch::empty({batch, 4 * n}, bytes);
  auto outgoing = torch::full({batch, 4 * n}, -1, indices);
  auto incoming = torch::full({batch, 4 * n}, -1, indices);
  auto modes = torch::zeros({batch, 4 * n}, bytes);
  auto vmask = torch::empty({batch, n}, ints), fmask = torch::empty_like(vmask);
  auto prefix = torch::empty({2, batch, n + 1}, indices);
  auto vp = prefix[0], fp = prefix[1];
  int64_t nv = vertex_capacity < 0 ? 0 : vertex_capacity;
  int64_t nf = face_capacity < 0 ? 0 : face_capacity;
  if (batch) {
    realize_kernel<index_t><<<GET_BLOCKS(batch * n, 256), 256, 0, stream>>>(
        static_cast<index_t>(batch * n),
        getTensorInfoCompact<bool, index_t, 3>(alive),
        getTensorInfoCompact<bool, index_t, 3>(right),
        getTensorInfoCompact<bool, index_t, 3>(down),
        getTensorInfoCompact<bool, index_t, 3>(er),
        getTensorInfoCompact<bool, index_t, 3>(ed));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "pixel_grid_mesh_corners", [&] {
      corner_kernel<scalar_t, index_t><<<GET_BLOCKS(batch * 4 * n, 256), 256, 0, stream>>>(
          static_cast<index_t>(batch * 4 * n),
          getTensorInfoCompact<bool, index_t, 3>(alive),
          getTensorInfoCompact<bool, index_t, 3>(er),
          getTensorInfoCompact<bool, index_t, 3>(ed),
          getTensorInfoCompact<scalar_t, index_t, 3>(depth),
          getTensorInfoCompact<scalar_t, index_t, 4>(displacement),
          getTensorInfoCompact<scalar_t, index_t, 3>(intrinsics),
          getTensorInfoCompact<index_t, index_t, 2>(owners),
          getTensorInfoCompact<scalar_t, index_t, 3>(core),
          getTensorInfoCompact<uint8_t, index_t, 2>(sizes));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    if (feather) {
      boundary_kernel<index_t><<<GET_BLOCKS(batch * 4 * n, 256), 256, 0, stream>>>(
          static_cast<index_t>(batch * 4 * n),
          getTensorInfoCompact<bool, index_t, 3>(alive),
          getTensorInfoCompact<bool, index_t, 3>(er),
          getTensorInfoCompact<bool, index_t, 3>(ed),
          getTensorInfoCompact<index_t, index_t, 2>(owners),
          getTensorInfoCompact<index_t, index_t, 2>(outgoing),
          getTensorInfoCompact<index_t, index_t, 2>(incoming),
          getTensorInfoCompact<uint8_t, index_t, 2>(modes));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "pixel_grid_mesh_modes", [&] {
        mode_kernel<scalar_t, index_t><<<GET_BLOCKS(batch * 4 * n, 256), 256, 0, stream>>>(
            static_cast<index_t>(batch * 4 * n),
            static_cast<index_t>(4 * n),
            getTensorInfoCompact<index_t, index_t, 2>(owners),
            getTensorInfoCompact<scalar_t, index_t, 3>(core),
            getTensorInfoCompact<index_t, index_t, 2>(outgoing),
            getTensorInfoCompact<index_t, index_t, 2>(incoming),
            getTensorInfoCompact<scalar_t, index_t, 3>(intrinsics),
            getTensorInfoCompact<scalar_t, index_t, 3>(width),
            getTensorInfoCompact<uint8_t, index_t, 2>(modes));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
    }
    count_kernel<index_t><<<GET_BLOCKS(batch * (n + 1), 256), 256, 0, stream>>>(
        static_cast<index_t>(batch * (n + 1)),
        static_cast<index_t>(n),
        getTensorInfoCompact<bool, index_t, 3>(alive),
        getTensorInfoCompact<index_t, index_t, 2>(owners),
        getTensorInfoCompact<index_t, index_t, 2>(outgoing),
        getTensorInfoCompact<uint8_t, index_t, 2>(modes),
        feather,
        getTensorInfoCompact<int32_t, index_t, 2>(vmask),
        getTensorInfoCompact<int32_t, index_t, 2>(fmask),
        getTensorInfoCompact<index_t, index_t, 2>(vp),
        getTensorInfoCompact<index_t, index_t, 2>(fp));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto cumulative = torch::empty_like(prefix);
    const int64_t count = prefix.numel();
    size_t temp_bytes = 0;
    C10_CUDA_CHECK(
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            temp_bytes,
            prefix.template data_ptr<index_t>(),
            cumulative.template data_ptr<index_t>(),
            count,
            stream));
    auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, bytes);
    C10_CUDA_CHECK(
        cub::DeviceScan::ExclusiveSum(
            temp.data_ptr(),
            temp_bytes,
            prefix.template data_ptr<index_t>(),
            cumulative.template data_ptr<index_t>(),
            count,
            stream));
    local_prefix_kernel<index_t><<<GET_BLOCKS(count, 256), 256, 0, stream>>>(
        static_cast<index_t>(count),
        getTensorInfoCompact<index_t, index_t, 3>(cumulative),
        getTensorInfoCompact<index_t, index_t, 3>(prefix),
        vertex_capacity,
        face_capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (vertex_capacity < 0) {
      auto maxima = prefix.select(2, n).amax(1).cpu();
      nv = maxima.template data_ptr<index_t>()[0];
      nf = maxima.template data_ptr<index_t>()[1];
    }
  }
  TORCH_CHECK(
      nv <= std::numeric_limits<int32_t>::max(),
      "pixel_grid_mesh(): a mesh has too many vertices for int32 face indices");
  auto v = torch::empty({batch, nv, 3}, depth.options());
  auto uv0 = torch::empty({batch, nv, 2}, depth.options()), uv1 = torch::empty_like(uv0);
  auto faces = torch::empty({batch, nf, 3}, ints);
  auto opacity =
      feather ? torch::empty({batch, nv}, depth.options()) : torch::empty({0, 0}, depth.options());
  auto cells = metadata ? torch::empty({batch, nf}, depth.options().dtype(at::kLong))
                        : torch::empty({0, 0}, depth.options().dtype(at::kLong));
  auto ids = torch::empty_like(cells);
  auto optional_width = feather ? width : width.view({0, 0, 0});
  auto optional_diagonal = diagonal.dim() == 3 ? diagonal : diagonal.view({0, 0, 0});
  AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "pixel_grid_mesh_emit", [&] {
    if (batch && nv) {
      vertex_kernel<scalar_t, index_t><<<GET_BLOCKS(batch * n, 256), 256, 0, stream>>>(
          static_cast<index_t>(batch * n),
          static_cast<index_t>(n),
          static_cast<index_t>(h),
          static_cast<index_t>(w),
          getTensorInfoCompact<index_t, index_t, 2>(owners),
          getTensorInfoCompact<scalar_t, index_t, 3>(core),
          getTensorInfoCompact<index_t, index_t, 2>(outgoing),
          getTensorInfoCompact<index_t, index_t, 2>(incoming),
          getTensorInfoCompact<uint8_t, index_t, 2>(modes),
          getTensorInfoCompact<int32_t, index_t, 2>(vmask),
          getTensorInfoCompact<index_t, index_t, 2>(vp),
          getTensorInfoCompact<scalar_t, index_t, 3>(intrinsics),
          getTensorInfoCompact<scalar_t, index_t, 3>(optional_width),
          getTensorInfoCompact<scalar_t, index_t, 3>(v),
          getTensorInfoCompact<scalar_t, index_t, 3>(uv0),
          getTensorInfoCompact<scalar_t, index_t, 3>(uv1),
          getTensorInfoCompact<scalar_t, index_t, 2>(opacity),
          static_cast<index_t>(nv));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (batch && nf) {
      face_kernel<scalar_t, index_t><<<GET_BLOCKS(batch * n, 256), 256, 0, stream>>>(
          static_cast<index_t>(batch * n),
          static_cast<index_t>(n),
          static_cast<index_t>(w),
          getTensorInfoCompact<index_t, index_t, 2>(owners),
          getTensorInfoCompact<scalar_t, index_t, 3>(core),
          getTensorInfoCompact<scalar_t, index_t, 3>(depth),
          getTensorInfoCompact<scalar_t, index_t, 4>(displacement),
          getTensorInfoCompact<scalar_t, index_t, 3>(intrinsics),
          getTensorInfoCompact<bool, index_t, 3>(optional_diagonal),
          getTensorInfoCompact<index_t, index_t, 2>(outgoing),
          getTensorInfoCompact<index_t, index_t, 2>(incoming),
          getTensorInfoCompact<uint8_t, index_t, 2>(modes),
          getTensorInfoCompact<int32_t, index_t, 2>(vmask),
          getTensorInfoCompact<index_t, index_t, 2>(vp),
          getTensorInfoCompact<int32_t, index_t, 2>(fmask),
          getTensorInfoCompact<index_t, index_t, 2>(fp),
          getTensorInfoCompact<int32_t, index_t, 3>(faces),
          getTensorInfoCompact<int64_t, index_t, 2>(cells),
          getTensorInfoCompact<int64_t, index_t, 2>(ids),
          static_cast<index_t>(nf));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });
  return {
      v,
      uv0,
      uv1,
      faces,
      opacity,
      cells,
      ids,
      owners,
      core,
      sizes,
      outgoing,
      incoming,
      modes,
      vmask,
      vp};
}

template <typename index_t>
std::vector<torch::Tensor> pixel_grid_mesh_backward_impl(
    const std::vector<torch::Tensor>& s,
    const torch::Tensor& grad_v,
    const torch::Tensor& grad_uv,
    bool grad_intrinsics,
    bool grad_width) {
  const auto &depth = s[0], &displacement = s[1], &k = s[2], &width = s[3], &v = s[4],
             &owners = s[5], &core = s[6], &sizes = s[7], &outgoing = s[8], &incoming = s[9],
             &modes = s[10], &mask = s[11], &prefix = s[12];
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t batch = depth.size(0), h = depth.size(1), w = depth.size(2), n = h * w,
                nv = v.size(1);
  const bool feather = width.dim() == 3;
  auto gd = torch::zeros(depth.sizes(), depth.options());
  auto gdelta = torch::zeros(displacement.sizes(), displacement.options());
  auto gk = torch::zeros(k.sizes(), k.options()), gw = torch::zeros(width.sizes(), width.options());
  if (!batch || !nv)
    return {gd, gdelta, gk, gw};
  auto grad = torch::empty_like(v);
  auto empty = torch::empty({0, 0, 0}, depth.options());
  auto gcore = feather ? torch::zeros_like(core) : empty;
  auto gv = grad_v.defined() ? grad_v.contiguous() : empty;
  auto guv = grad_uv.defined() ? grad_uv.contiguous() : empty;
  auto optional_gk = grad_intrinsics ? gk : empty;
  auto optional_gw = grad_width ? gw : empty;
  AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "pixel_grid_mesh_backward", [&] {
    projection_backward_kernel<scalar_t, index_t><<<GET_BLOCKS(batch * nv, 256), 256, 0, stream>>>(
        static_cast<index_t>(batch * nv),
        static_cast<index_t>(nv),
        static_cast<index_t>(h),
        static_cast<index_t>(w),
        getTensorInfoCompact<scalar_t, index_t, 3>(v),
        getTensorInfoCompact<scalar_t, index_t, 3>(k),
        getTensorInfoCompact<scalar_t, index_t, 3>(gv),
        getTensorInfoCompact<scalar_t, index_t, 3>(guv),
        getTensorInfoCompact<scalar_t, index_t, 3>(grad),
        getTensorInfoCompact<scalar_t, index_t, 3>(optional_gk),
        getTensorInfoCompact<index_t, index_t, 2>(prefix));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (feather) {
      feather_backward_kernel<scalar_t, index_t>
          <<<GET_BLOCKS(batch * 16 * n, 256), 256, 0, stream>>>(
              static_cast<index_t>(batch * 16 * n),
              static_cast<index_t>(n),
              getTensorInfoCompact<index_t, index_t, 2>(owners),
              getTensorInfoCompact<scalar_t, index_t, 3>(core),
              getTensorInfoCompact<index_t, index_t, 2>(outgoing),
              getTensorInfoCompact<index_t, index_t, 2>(incoming),
              getTensorInfoCompact<uint8_t, index_t, 2>(modes),
              getTensorInfoCompact<int32_t, index_t, 2>(mask),
              getTensorInfoCompact<index_t, index_t, 2>(prefix),
              getTensorInfoCompact<scalar_t, index_t, 3>(k),
              getTensorInfoCompact<scalar_t, index_t, 3>(width),
              getTensorInfoCompact<scalar_t, index_t, 3>(grad),
              getTensorInfoCompact<scalar_t, index_t, 3>(gcore),
              getTensorInfoCompact<scalar_t, index_t, 3>(optional_gw),
              getTensorInfoCompact<scalar_t, index_t, 3>(optional_gk));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    corner_backward_kernel<scalar_t, index_t><<<GET_BLOCKS(batch * n, 256), 256, 0, stream>>>(
        static_cast<index_t>(batch * n),
        static_cast<index_t>(n),
        static_cast<index_t>(w),
        getTensorInfoCompact<scalar_t, index_t, 3>(depth),
        getTensorInfoCompact<scalar_t, index_t, 3>(k),
        getTensorInfoCompact<index_t, index_t, 2>(owners),
        getTensorInfoCompact<uint8_t, index_t, 2>(sizes),
        getTensorInfoCompact<int32_t, index_t, 2>(mask),
        getTensorInfoCompact<index_t, index_t, 2>(prefix),
        getTensorInfoCompact<scalar_t, index_t, 3>(grad),
        getTensorInfoCompact<scalar_t, index_t, 3>(gcore),
        getTensorInfoCompact<scalar_t, index_t, 3>(gd),
        getTensorInfoCompact<scalar_t, index_t, 4>(gdelta),
        getTensorInfoCompact<scalar_t, index_t, 3>(optional_gk));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {gd, gdelta, gk, gw};
}

void check_tensor(
    const torch::Tensor& tensor,
    const torch::Tensor& depth,
    at::ScalarType type,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == depth.device() && tensor.scalar_type() == type &&
          tensor.layout() == torch::kStrided,
      "pixel_grid_mesh(): expected ",
      name,
      " on ",
      depth.device(),
      " with dtype ",
      type,
      " and strided layout");
}

} // namespace

std::vector<torch::Tensor> pixel_grid_mesh_forward(
    const torch::Tensor& depth,
    const torch::Tensor& displacement,
    const torch::Tensor& intrinsics,
    const torch::Tensor& alive,
    const torch::Tensor& right,
    const torch::Tensor& down,
    const torch::Tensor& diagonal,
    const torch::Tensor& width,
    bool metadata,
    int64_t vertex_capacity,
    int64_t face_capacity) {
  TORCH_CHECK(
      depth.is_cuda(), "pixel_grid_mesh(): expected CUDA tensors; use pixel_grid_mesh_ref for CPU");
  TORCH_CHECK(
      depth.dim() == 3 && depth.size(1) > 0 && depth.size(2) > 0,
      "pixel_grid_mesh(): expected depth shape [N, H, W] with H, W > 0, got ",
      depth.sizes());
  TORCH_CHECK(
      depth.scalar_type() == at::kFloat || depth.scalar_type() == at::kDouble,
      "pixel_grid_mesh(): expected float32 or float64 depth, got ",
      depth.dtype());
  const int64_t batch = depth.size(0), h = depth.size(1), w = depth.size(2);
  check_tensor(depth, depth, depth.scalar_type(), "depth");
  check_tensor(displacement, depth, depth.scalar_type(), "displacement");
  check_tensor(intrinsics, depth, depth.scalar_type(), "intrinsics");
  check_tensor(alive, depth, at::kBool, "alive");
  check_tensor(right, depth, at::kBool, "join_right");
  check_tensor(down, depth, at::kBool, "join_down");
  check_tensor(diagonal, depth, at::kBool, "diagonal");
  check_tensor(width, depth, depth.scalar_type(), "feather_width");
  TORCH_CHECK(
      displacement.sizes() == at::IntArrayRef({batch, h, w, 3}),
      "pixel_grid_mesh(): expected displacement shape [N, H, W, 3], got ",
      displacement.sizes());
  TORCH_CHECK(
      intrinsics.sizes() == at::IntArrayRef({batch, 3, 3}),
      "pixel_grid_mesh(): expected intrinsics shape [N, 3, 3], got ",
      intrinsics.sizes());
  TORCH_CHECK(
      alive.sizes() == depth.sizes(),
      "pixel_grid_mesh(): expected alive shape [N, H, W], got ",
      alive.sizes());
  TORCH_CHECK(
      right.sizes() == at::IntArrayRef({batch, h, w - 1}),
      "pixel_grid_mesh(): expected join_right shape [N, H, W-1], got ",
      right.sizes());
  TORCH_CHECK(
      down.sizes() == at::IntArrayRef({batch, h - 1, w}),
      "pixel_grid_mesh(): expected join_down shape [N, H-1, W], got ",
      down.sizes());
  TORCH_CHECK(
      (diagonal.dim() == 1 && diagonal.numel() == 0) || diagonal.sizes() == depth.sizes(),
      "pixel_grid_mesh(): expected diagonal shape [N, H, W], got ",
      diagonal.sizes());
  TORCH_CHECK(
      (width.dim() == 1 && width.numel() == 0) || width.sizes() == depth.sizes(),
      "pixel_grid_mesh(): expected feather_width shape [N, H, W], got ",
      width.sizes());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(depth));
  TORCH_CHECK(
      (vertex_capacity == -1 && face_capacity == -1) ||
          (vertex_capacity >= 0 && face_capacity >= 0 &&
           vertex_capacity <= std::numeric_limits<int32_t>::max()),
      "pixel_grid_mesh(): output capacities must both be nonnegative, "
      "with at most int32 vertices, or both -1");
  // Account for the largest padded output and the combined vertex/face scan.
  bool index32 = depth.numel() <= std::numeric_limits<int32_t>::max() / 64 &&
      h <= std::numeric_limits<int32_t>::max() / 64 / w;
  if (vertex_capacity >= 0 && batch > 0) {
    index32 = index32 && vertex_capacity <= std::numeric_limits<int32_t>::max() / 3 / batch &&
        face_capacity <= std::numeric_limits<int32_t>::max() / 3 / batch;
  }
  for (const auto& tensor : {depth, displacement, intrinsics, alive, right, down, diagonal, width})
    index32 = index32 && at::native::canUse32BitIndexMath(tensor);
  if (index32)
    return pixel_grid_mesh_forward_impl<int32_t>(
        depth,
        displacement,
        intrinsics,
        alive,
        right,
        down,
        diagonal,
        width,
        metadata,
        vertex_capacity,
        face_capacity);
  return pixel_grid_mesh_forward_impl<int64_t>(
      depth,
      displacement,
      intrinsics,
      alive,
      right,
      down,
      diagonal,
      width,
      metadata,
      vertex_capacity,
      face_capacity);
}

std::vector<torch::Tensor> pixel_grid_mesh_backward(
    const std::vector<torch::Tensor>& saved,
    const torch::Tensor& grad_v,
    const torch::Tensor& grad_uv,
    bool grad_intrinsics,
    bool grad_width) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(saved[0]));
  if (saved[5].scalar_type() == at::kInt)
    return pixel_grid_mesh_backward_impl<int32_t>(
        saved, grad_v, grad_uv, grad_intrinsics, grad_width);
  return pixel_grid_mesh_backward_impl<int64_t>(
      saved, grad_v, grad_uv, grad_intrinsics, grad_width);
}

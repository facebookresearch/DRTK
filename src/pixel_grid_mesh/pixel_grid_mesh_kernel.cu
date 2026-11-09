// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license in the root directory.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <limits>
#include "pixel_grid_mesh_kernel.h"

namespace {
constexpr int kThreads = 256;

template <typename T>
struct V2 {
  T x, y;
};
template <typename T>
struct V3 {
  T x, y, z;
};
template <typename T>
__device__ V2<T> operator+(V2<T> a, V2<T> b) {
  return {a.x + b.x, a.y + b.y};
}
template <typename T>
__device__ V2<T> operator-(V2<T> a, V2<T> b) {
  return {a.x - b.x, a.y - b.y};
}
template <typename T>
__device__ V2<T> operator*(V2<T> a, T b) {
  return {a.x * b, a.y * b};
}
template <typename T>
__device__ V3<T> operator+(V3<T> a, V3<T> b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template <typename T>
__device__ V3<T> operator*(V3<T> a, T b) {
  return {a.x * b, a.y * b, a.z * b};
}
template <typename T>
__device__ T dot(V2<T> a, V2<T> b) {
  return a.x * b.x + a.y * b.y;
}
template <typename T>
__device__ T dot(V3<T> a, V3<T> b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T>
__device__ V3<T> read3(const T* p, int i) {
  return {p[3 * i], p[3 * i + 1], p[3 * i + 2]};
}
template <typename T>
__device__ void write3(T* p, int i, V3<T> v) {
  p[3 * i] = v.x;
  p[3 * i + 1] = v.y;
  p[3 * i + 2] = v.z;
}
template <typename T>
__device__ void add3(T* p, int i, V3<T> v) {
  atomicAdd(p + 3 * i, v.x);
  atomicAdd(p + 3 * i + 1, v.y);
  atomicAdd(p + 3 * i + 2, v.z);
}
template <typename T>
__device__ V3<T> mul(const T* m, V3<T> v) {
  return {
      m[0] * v.x + m[1] * v.y + m[2] * v.z,
      m[3] * v.x + m[4] * v.y + m[5] * v.z,
      m[6] * v.x + m[7] * v.y + m[8] * v.z};
}
template <typename T>
__device__ V3<T> mult(const T* m, V3<T> v) {
  return {
      m[0] * v.x + m[3] * v.y + m[6] * v.z,
      m[1] * v.x + m[4] * v.y + m[7] * v.z,
      m[2] * v.x + m[5] * v.y + m[8] * v.z};
}
template <typename T>
__device__ void outer_add(T* g, V3<T> a, V3<T> b) {
  if (!g)
    return;
  const T av[3] = {a.x, a.y, a.z}, bv[3] = {b.x, b.y, b.z};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      atomicAdd(g + 3 * i + j, av[i] * bv[j]);
}
template <typename T>
__device__ void inverse3(const T* m, T* a) {
  a[0] = m[4] * m[8] - m[5] * m[7];
  a[1] = m[2] * m[7] - m[1] * m[8];
  a[2] = m[1] * m[5] - m[2] * m[4];
  a[3] = m[5] * m[6] - m[3] * m[8];
  a[4] = m[0] * m[8] - m[2] * m[6];
  a[5] = m[2] * m[3] - m[0] * m[5];
  a[6] = m[3] * m[7] - m[4] * m[6];
  a[7] = m[1] * m[6] - m[0] * m[7];
  a[8] = m[0] * m[4] - m[1] * m[3];
  const T det = m[0] * a[0] + m[1] * a[3] + m[2] * a[6];
  for (int i = 0; i < 9; ++i)
    a[i] /= det;
}
template <typename T>
__device__ V2<T> project(const T* k, V3<T> v) {
  const auto h = mul(k, v);
  return {h.x / h.z, h.y / h.z};
}
template <typename T>
__device__ V3<T> project_backward(const T* k, V3<T> v, V2<T> g, T* gk) {
  const auto h = mul(k, v);
  const V3<T> gh = {g.x / h.z, g.y / h.z, -(g.x * h.x + g.y * h.y) / (h.z * h.z)};
  outer_add(gk, gh, v);
  return mult(k, gh);
}
template <typename T>
__device__ V2<T> unit(V2<T> v) {
  const T len = sqrt(dot(v, v));
  const T d = len > std::numeric_limits<T>::epsilon() ? len : std::numeric_limits<T>::epsilon();
  return v * (T(1) / d);
}
template <typename T>
__device__ V2<T> unit_backward(V2<T> v, V2<T> g) {
  const T len = sqrt(dot(v, v)), eps = std::numeric_limits<T>::epsilon();
  if (len <= eps)
    return g * (T(1) / eps);
  const auto t = v * (T(1) / len);
  return (g - t * dot(g, t)) * (T(1) / len);
}

struct Grid {
  int h, w;
  const bool *alive, *right, *down;
  __device__ bool live(int y, int x) const {
    return y >= 0 && y < h && x >= 0 && x < w && alive[y * w + x];
  }
  __device__ bool cut_r(int y, int x) const {
    return live(y, x) && live(y, x + 1) && !right[y * (w - 1) + x];
  }
  __device__ bool cut_d(int y, int x) const {
    return live(y, x) && live(y + 1, x) && !down[y * w + x];
  }
  __device__ bool supported(int y, int x) const {
    int incident =
        int(live(y - 1, x - 1)) + int(live(y - 1, x)) + int(live(y, x - 1)) + int(live(y, x));
    int degree = int(cut_r(y - 1, x - 1)) + int(cut_r(y, x - 1)) + int(cut_d(y - 1, x - 1)) +
        int(cut_d(y - 1, x));
    return degree > 1 || (incident > 0 && incident < 4);
  }
  __device__ bool joined_r(int y, int x) const {
    return live(y, x) && live(y, x + 1) && right[y * (w - 1) + x];
  }
  __device__ bool joined_d(int y, int x) const {
    return live(y, x) && live(y + 1, x) && down[y * w + x];
  }
};

__global__ void realize_kernel(Grid g, bool* right, bool* down) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= g.h * g.w)
    return;
  int y = c / g.w, x = c % g.w;
  if (x + 1 < g.w)
    right[y * (g.w - 1) + x] = g.live(y, x) && g.live(y, x + 1) &&
        (!g.cut_r(y, x) || !(g.supported(y, x + 1) || g.supported(y + 1, x + 1)));
  if (y + 1 < g.h)
    down[y * g.w + x] = g.live(y, x) && g.live(y + 1, x) &&
        (!g.cut_d(y, x) || !(g.supported(y + 1, x) || g.supported(y + 1, x + 1)));
}

__device__ void lattice_corner(int raw, int w, int& y, int& x) {
  const int c = raw / 4, s = raw % 4;
  y = c / w + (s >= 2);
  x = c % w + (s == 1 || s == 2);
}

template <typename T>
__global__ void corner_kernel(
    Grid g,
    const T* d,
    const T* displacement,
    const T* k,
    int* owners,
    T* core,
    uint8_t* sizes) {
  int raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= 4 * g.h * g.w)
    return;
  if (!g.alive[raw / 4]) {
    owners[raw] = -1;
    sizes[raw] = 0;
    return;
  }
  int y, x;
  lattice_corner(raw, g.w, y, x);
  const int cy[4] = {y - 1, y - 1, y, y}, cx[4] = {x - 1, x, x, x - 1}, local[4] = {2, 3, 0, 1};
  int ids[4], label[4], me = 0;
  for (int i = 0; i < 4; ++i) {
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
  for (int step = 0; step < 3; ++step) {
    int next[4];
    for (int i = 0; i < 4; ++i) {
      next[i] = label[i];
      if (link[i])
        next[i] = min(next[i], label[(i + 1) % 4]);
      if (link[(i + 3) % 4])
        next[i] = min(next[i], label[(i + 3) % 4]);
    }
    for (int i = 0; i < 4; ++i)
      label[i] = next[i];
  }
  int owner = raw, count = 0;
  for (int i = 0; i < 4; ++i)
    if (ids[i] >= 0 && label[i] == label[me]) {
      owner = min(owner, ids[i]);
      ++count;
    }
  owners[raw] = owner;
  sizes[raw] = count;
  if (owner != raw)
    return;
  T inv[9];
  inverse3(k, inv);
  const auto ray = mul(inv, V3<T>{T(x) - T(.5), T(y) - T(.5), T(1)});
  V3<T> v = {0, 0, 0};
  for (int i = 0; i < 4; ++i)
    if (ids[i] >= 0 && label[i] == label[me]) {
      int c = ids[i] / 4;
      v = v + ray * d[c] + read3(displacement, c);
    }
  write3(core, raw, v * (T(1) / T(count)));
}

__global__ void
boundary_kernel(Grid g, const int* owners, int* outgoing, int* incoming, uint8_t* modes) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= 4 * g.h * g.w)
    return;
  int c = e / 4, s = e % 4, y = c / g.w, x = c % g.w;
  bool joined = s == 0
      ? g.joined_d(y - 1, x)
      : (s == 1 ? g.joined_r(y, x) : (s == 2 ? g.joined_d(y, x) : g.joined_r(y, x - 1)));
  bool boundary = g.alive[c] && !joined;
  modes[e] = boundary ? 1 : 0;
  if (boundary) {
    outgoing[owners[e]] = e;
    incoming[owners[4 * c + (s + 1) % 4]] = e;
  }
}

template <typename T>
struct EdgeGeometry {
  int a, b, p, previous, following;
  V2<T> tangent, previous_tangent, normal, previous_normal, miter;
  T width, denominator;
};

template <typename T>
__device__ EdgeGeometry<T> edge_geometry(
    int e,
    const int* owners,
    const T* core,
    const int* outgoing,
    const int* incoming,
    const T* k,
    const T* width) {
  EdgeGeometry<T> r;
  r.a = owners[e];
  r.b = owners[4 * (e / 4) + (e % 4 + 1) % 4];
  r.previous = incoming[r.a];
  r.following = outgoing[r.b];
  r.p = owners[r.previous];
  const auto a = project(k, read3(core, r.a)), b = project(k, read3(core, r.b)),
             p = project(k, read3(core, r.p));
  r.tangent = unit(b - a);
  r.previous_tangent = unit(a - p);
  r.normal = {r.tangent.y, -r.tangent.x};
  r.previous_normal = {r.previous_tangent.y, -r.previous_tangent.x};
  r.denominator = T(1) + dot(r.normal, r.previous_normal);
  T denom = r.denominator > std::numeric_limits<T>::epsilon() ? r.denominator
                                                              : std::numeric_limits<T>::epsilon();
  r.miter = (r.normal + r.previous_normal) * (T(1) / denom);
  r.width = T(.5) * (width[e / 4] + width[r.previous / 4]);
  return r;
}

template <typename T>
__global__ void mode_kernel(
    int edges,
    const int* owners,
    const T* core,
    const int* outgoing,
    const int* incoming,
    const T* k,
    const T* width,
    uint8_t* modes) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= edges || modes[e] == 0)
    return;
  const auto r = edge_geometry(e, owners, core, outgoing, incoming, k, width);
  modes[e] = r.denominator < T(1e-4) ? 3 : (dot(r.miter, r.miter) > T(16) ? 2 : 1);
}

__global__ void count_kernel(
    int n,
    const bool* alive,
    const int* owners,
    const int* outgoing,
    const uint8_t* modes,
    bool feather,
    int* vmask,
    int* fmask,
    int* vcount,
    int* fcount) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c > n)
    return;
  if (c == n) {
    vcount[c] = fcount[c] = 0;
    return;
  }
  unsigned v = 0, f = alive[c] ? 3 : 0;
  for (int s = 0; s < 4; ++s) {
    int e = 4 * c + s;
    if (owners[e] == e)
      v |= 1u << s;
    if (feather && modes[e]) {
      v |= 1u << (4 + s);
      f |= 3u << (2 + 4 * s);
      int next = outgoing[owners[4 * c + (s + 1) % 4]];
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

__device__ int compact_vertex(int cell, int slot, const int* mask, const int* prefix) {
  return prefix[cell] + __popc(static_cast<unsigned>(mask[cell]) & ((1u << slot) - 1));
}
__device__ int core_vertex(int raw, const int* mask, const int* prefix) {
  return compact_vertex(raw / 4, raw % 4, mask, prefix);
}
__device__ int outer_first(int e, const int* mask, const int* prefix) {
  return compact_vertex(e / 4, 4 + e % 4, mask, prefix);
}
__device__ int outer_last(
    int e,
    const int* owners,
    const int* outgoing,
    const uint8_t* modes,
    const int* mask,
    const int* prefix) {
  int next = outgoing[owners[4 * (e / 4) + (e % 4 + 1) % 4]];
  return modes[next] >= 2 ? compact_vertex(e / 4, 8 + e % 4, mask, prefix)
                          : outer_first(next, mask, prefix);
}

template <typename T>
__global__ void vertex_kernel(
    int n,
    int h,
    int w,
    const int* owners,
    const T* core,
    const int* outgoing,
    const int* incoming,
    const uint8_t* modes,
    const int* mask,
    const int* prefix,
    const T* k,
    const T* width,
    T* v,
    T* uv0,
    T* uv1,
    T* opacity) {
  int raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= 16 * n)
    return;
  int c = raw / 16, s = raw % 16;
  if (!(static_cast<unsigned>(mask[c]) & (1u << s)))
    return;
  int index = compact_vertex(c, s, mask, prefix), anchor = 4 * c + s;
  V3<T> value;
  if (s < 4)
    value = read3(core, anchor);
  else {
    int e = 4 * c + s % 4;
    auto r = edge_geometry(e, owners, core, outgoing, incoming, k, width);
    V2<T> offset;
    anchor = r.a;
    if (s < 8)
      offset = (modes[e] >= 2 ? r.normal : r.miter) * r.width;
    else if (s < 12) {
      anchor = r.b;
      T wid = T(.5) * (width[c] + width[r.following / 4]);
      offset = r.normal * wid;
    } else
      offset = r.previous_tangent * r.width;
    T inv[9];
    inverse3(k, inv);
    value = read3(core, anchor);
    value = value + mul(inv, V3<T>{offset.x, offset.y, T(0)}) * value.z;
  }
  write3(v, index, value);
  int y, x;
  lattice_corner(anchor, w, y, x);
  uv0[2 * index] = T(x) / T(w);
  uv0[2 * index + 1] = T(y) / T(h);
  const auto pixel = project(k, value);
  uv1[2 * index] = (pixel.x + T(.5)) / T(w);
  uv1[2 * index + 1] = (pixel.y + T(.5)) / T(h);
  if (opacity)
    opacity[index] = s < 4 ? T(1) : T(0);
}

template <typename T>
__global__ void face_kernel(
    int n,
    int w,
    const int* owners,
    const T* core,
    const T* depth,
    const T* displacement,
    const T* k,
    const bool* diagonal,
    const int* outgoing,
    const int* incoming,
    const uint8_t* modes,
    const int* vmask,
    const int* vp,
    const int* fmask,
    const int* fp,
    int* faces,
    int64_t* cells,
    int64_t* ids) {
  int raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= 18 * n)
    return;
  int c = raw / 18, s = raw % 18;
  if (!(static_cast<unsigned>(fmask[c]) & (1u << s)))
    return;
  int f = fp[c] + __popc(static_cast<unsigned>(fmask[c]) & ((1u << s) - 1));
  int a, b, d, id;
  if (s < 2) {
    int q[4];
    for (int i = 0; i < 4; ++i)
      q[i] = owners[4 * c + i];
    bool chosen;
    if (diagonal)
      chosen = diagonal[c];
    else {
      T inv[9];
      inverse3(k, inv);
      const auto target =
          mul(inv, V3<T>{T(c % w), T(c / w), T(1)}) * depth[c] + read3(displacement, c);
      auto d02 = (read3(core, q[0]) + read3(core, q[2])) * T(.5) + target * T(-1);
      auto d13 = (read3(core, q[1]) + read3(core, q[3])) * T(.5) + target * T(-1);
      chosen = dot(d13, d13) < dot(d02, d02);
    }
    const int pattern0[6] = {0, 1, 2, 0, 2, 3}, pattern1[6] = {0, 1, 3, 1, 2, 3};
    const int* pat = chosen ? pattern1 : pattern0;
    a = core_vertex(q[pat[3 * s]], vmask, vp);
    b = core_vertex(q[pat[3 * s + 1]], vmask, vp);
    d = core_vertex(q[pat[3 * s + 2]], vmask, vp);
    id = 32 * c + s;
  } else {
    int side = (s - 2) / 4, part = (s - 2) % 4, e = 4 * c + side;
    int start = owners[e], end = owners[4 * c + (side + 1) % 4],
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
      int prev = incoming[start];
      a = inner;
      if (modes[e] == 2) {
        b = outer_last(prev, owners, outgoing, modes, vmask, vp);
        d = first;
      } else {
        int cap = compact_vertex(c, 12 + side, vmask, vp);
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

template <typename T>
__global__ void projection_backward_kernel(
    int vertices,
    int h,
    int w,
    const T* v,
    const T* k,
    const T* gv,
    const T* guv,
    T* result,
    T* gk) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertices)
    return;
  V3<T> g = gv ? read3(gv, i) : V3<T>{0, 0, 0};
  if (guv)
    g = g + project_backward(k, read3(v, i), V2<T>{guv[2 * i] / T(w), guv[2 * i + 1] / T(h)}, gk);
  write3(result, i, g);
}

template <typename T>
__device__ void
tangent_backward(int a, int b, V2<T> gradient, const T* core, const T* k, T* gcore, T* gk) {
  const auto pa = project(k, read3(core, a)), pb = project(k, read3(core, b));
  auto edge_grad = unit_backward(pb - pa, gradient);
  add3(gcore, a, project_backward(k, read3(core, a), edge_grad * T(-1), gk));
  add3(gcore, b, project_backward(k, read3(core, b), edge_grad, gk));
}

template <typename T>
__global__ void feather_backward_kernel(
    int n,
    const int* owners,
    const T* core,
    const int* outgoing,
    const int* incoming,
    const uint8_t* modes,
    const int* mask,
    const int* prefix,
    const T* k,
    const T* width,
    const T* grad,
    T* gcore,
    T* gwidth,
    T* gk) {
  int raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= 16 * n)
    return;
  int c = raw / 16, s = raw % 16;
  if (s < 4 || !(static_cast<unsigned>(mask[c]) & (1u << s)))
    return;
  int e = 4 * c + s % 4;
  auto r = edge_geometry(e, owners, core, outgoing, incoming, k, width);
  int anchor = r.a, other = r.previous / 4;
  T wid = r.width;
  V2<T> direction;
  if (s < 8)
    direction = modes[e] >= 2 ? r.normal : r.miter;
  else if (s < 12) {
    anchor = r.b;
    other = r.following / 4;
    wid = T(.5) * (width[c] + width[other]);
    direction = r.normal;
  } else
    direction = r.previous_tangent;
  const auto g = read3(grad, compact_vertex(c, s, mask, prefix));
  const auto pos = read3(core, anchor);
  T inv[9];
  inverse3(k, inv);
  const auto ray = mul(inv, V3<T>{direction.x * wid, direction.y * wid, T(0)});
  add3(gcore, anchor, V3<T>{g.x, g.y, g.z + dot(g, ray)});
  const auto gdelta = mult(inv, g) * pos.z;
  outer_add(gk, gdelta * T(-1), ray);
  const V2<T> gd = {gdelta.x, gdelta.y};
  const T gw = dot(gd, direction) * T(.5);
  if (gwidth) {
    atomicAdd(gwidth + c, gw);
    atomicAdd(gwidth + other, gw);
  }
  const auto gdir = gd * wid;
  if (s >= 12) {
    tangent_backward(r.p, r.a, gdir, core, k, gcore, gk);
    return;
  }
  V2<T> gn = gdir, gp = {0, 0};
  if (s < 8 && modes[e] == 1) {
    const T denom = r.denominator,
            coeff = dot(gdir, r.normal + r.previous_normal) / (denom * denom);
    gn = gdir * (T(1) / denom) - r.previous_normal * coeff;
    gp = gdir * (T(1) / denom) - r.normal * coeff;
  }
  tangent_backward(r.a, r.b, V2<T>{-gn.y, gn.x}, core, k, gcore, gk);
  if (s < 8 && modes[e] == 1)
    tangent_backward(r.p, r.a, V2<T>{-gp.y, gp.x}, core, k, gcore, gk);
}

template <typename T>
__global__ void corner_backward_kernel(
    int n,
    int w,
    const T* depth,
    const T* k,
    const int* owners,
    const uint8_t* sizes,
    const int* mask,
    const int* prefix,
    const T* grad,
    const T* gcore,
    T* gd,
    T* gdelta,
    T* gk) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n)
    return;
  T d = 0;
  V3<T> delta = {0, 0, 0};
  T inv[9];
  inverse3(k, inv);
  for (int s = 0; s < 4; ++s) {
    int raw = 4 * c + s, owner = owners[raw];
    if (owner < 0)
      continue;
    auto g = read3(grad, core_vertex(owner, mask, prefix));
    if (gcore)
      g = g + read3(gcore, owner);
    g = g * (T(1) / T(sizes[raw]));
    int y, x;
    lattice_corner(raw, w, y, x);
    const auto ray = mul(inv, V3<T>{T(x) - T(.5), T(y) - T(.5), T(1)});
    d += dot(g, ray);
    delta = delta + g;
    if (gk)
      outer_add(gk, mult(inv, g) * (T(-1) * depth[c]), ray);
  }
  gd[c] = d;
  write3(gdelta, c, delta);
}

int blocks(int n) {
  return (n + kThreads - 1) / kThreads;
}
void check_tensor(
    const torch::Tensor& t,
    const torch::Tensor& base,
    at::ScalarType type,
    const char* name) {
  TORCH_CHECK(
      t.device() == base.device() && t.scalar_type() == type && t.is_contiguous(),
      "pixel_grid_mesh: ",
      name,
      " must have the expected device/dtype and be contiguous");
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
    bool metadata) {
  TORCH_CHECK(depth.is_cuda(), "pixel_grid_mesh requires CUDA; use pixel_grid_mesh_ref for CPU");
  TORCH_CHECK(
      depth.dim() == 2 && depth.size(0) > 0 && depth.size(1) > 0,
      "depth must have nonempty shape [H,W]");
  TORCH_CHECK(
      depth.scalar_type() == at::kFloat || depth.scalar_type() == at::kDouble,
      "depth must be float32/float64");
  TORCH_CHECK(
      depth.numel() < std::numeric_limits<int>::max() / 32, "pixel grid exceeds int32 indexing");
  int h = depth.size(0), w = depth.size(1), n = h * w;
  bool feather = width.numel() != 0;
  check_tensor(depth, depth, depth.scalar_type(), "depth");
  check_tensor(displacement, depth, depth.scalar_type(), "displacement");
  check_tensor(intrinsics, depth, depth.scalar_type(), "intrinsics");
  check_tensor(alive, depth, at::kBool, "alive");
  check_tensor(right, depth, at::kBool, "right");
  check_tensor(down, depth, at::kBool, "down");
  check_tensor(diagonal, depth, at::kBool, "diagonal");
  check_tensor(width, depth, depth.scalar_type(), "width");
  TORCH_CHECK(
      displacement.sizes() == at::IntArrayRef({h, w, 3}) &&
          intrinsics.sizes() == at::IntArrayRef({3, 3}),
      "invalid displacement/intrinsics shape");
  TORCH_CHECK(
      alive.sizes() == depth.sizes() && right.sizes() == at::IntArrayRef({h, w - 1}) &&
          down.sizes() == at::IntArrayRef({h - 1, w}),
      "invalid topology shape");
  TORCH_CHECK(
      (diagonal.numel() == 0 || diagonal.sizes() == depth.sizes()) &&
          (!feather || width.sizes() == depth.sizes()),
      "invalid diagonal/width shape");
  const c10::cuda::CUDAGuard guard(depth.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto ints = depth.options().dtype(at::kInt), bytes = depth.options().dtype(at::kByte);
  auto er = torch::empty_like(right), ed = torch::empty_like(down);
  Grid input{h, w, alive.data_ptr<bool>(), right.data_ptr<bool>(), down.data_ptr<bool>()};
  realize_kernel<<<blocks(n), kThreads, 0, stream>>>(
      input, er.data_ptr<bool>(), ed.data_ptr<bool>());
  Grid effective{h, w, alive.data_ptr<bool>(), er.data_ptr<bool>(), ed.data_ptr<bool>()};
  auto owners = torch::empty({4 * n}, ints), core = torch::zeros({4 * n, 3}, depth.options()),
       sizes = torch::empty({4 * n}, bytes);
  auto outgoing = torch::full({4 * n}, -1, ints), incoming = torch::full({4 * n}, -1, ints),
       modes = torch::zeros({4 * n}, bytes);
  AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "pixel_grid_mesh_corners", [&] {
    corner_kernel<<<blocks(4 * n), kThreads, 0, stream>>>(
        effective,
        depth.data_ptr<scalar_t>(),
        displacement.data_ptr<scalar_t>(),
        intrinsics.data_ptr<scalar_t>(),
        owners.data_ptr<int>(),
        core.data_ptr<scalar_t>(),
        sizes.data_ptr<uint8_t>());
  });
  if (feather) {
    boundary_kernel<<<blocks(4 * n), kThreads, 0, stream>>>(
        effective,
        owners.data_ptr<int>(),
        outgoing.data_ptr<int>(),
        incoming.data_ptr<int>(),
        modes.data_ptr<uint8_t>());
    AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "pixel_grid_mesh_modes", [&] {
      mode_kernel<<<blocks(4 * n), kThreads, 0, stream>>>(
          4 * n,
          owners.data_ptr<int>(),
          core.data_ptr<scalar_t>(),
          outgoing.data_ptr<int>(),
          incoming.data_ptr<int>(),
          intrinsics.data_ptr<scalar_t>(),
          width.data_ptr<scalar_t>(),
          modes.data_ptr<uint8_t>());
    });
  }
  auto vmask = torch::empty({n}, ints), fmask = torch::empty({n}, ints),
       count = torch::empty({2, n + 1}, ints), prefix = torch::empty_like(count);
  count_kernel<<<blocks(n + 1), kThreads, 0, stream>>>(
      n,
      alive.data_ptr<bool>(),
      owners.data_ptr<int>(),
      outgoing.data_ptr<int>(),
      modes.data_ptr<uint8_t>(),
      feather,
      vmask.data_ptr<int>(),
      fmask.data_ptr<int>(),
      count.data_ptr<int>(),
      count.data_ptr<int>() + n + 1);
  size_t temp_bytes = 0;
  C10_CUDA_CHECK(
      cub::DeviceScan::ExclusiveSum(
          nullptr, temp_bytes, count.data_ptr<int>(), prefix.data_ptr<int>(), n + 1, stream));
  auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, bytes);
  for (int row = 0; row < 2; ++row)
    C10_CUDA_CHECK(
        cub::DeviceScan::ExclusiveSum(
            temp.data_ptr(),
            temp_bytes,
            count.data_ptr<int>() + row * (n + 1),
            prefix.data_ptr<int>() + row * (n + 1),
            n + 1,
            stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // One count transfer determines compact output sizes; there is no per-cell host work.
  auto totals = prefix.select(1, n).contiguous().cpu();
  int nv = totals.data_ptr<int>()[0], nf = totals.data_ptr<int>()[1];
  auto v = torch::empty({nv, 3}, depth.options()), uv0 = torch::empty({nv, 2}, depth.options()),
       uv1 = torch::empty_like(uv0);
  auto faces = torch::empty({nf, 3}, ints),
       opacity = torch::empty({feather ? nv : 0}, depth.options());
  auto cells = torch::empty({metadata ? nf : 0}, depth.options().dtype(at::kLong)),
       ids = torch::empty_like(cells);
  auto vp = prefix[0], fp = prefix[1];
  AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "pixel_grid_mesh_emit", [&] {
    vertex_kernel<<<blocks(16 * n), kThreads, 0, stream>>>(
        n,
        h,
        w,
        owners.data_ptr<int>(),
        core.data_ptr<scalar_t>(),
        outgoing.data_ptr<int>(),
        incoming.data_ptr<int>(),
        modes.data_ptr<uint8_t>(),
        vmask.data_ptr<int>(),
        vp.data_ptr<int>(),
        intrinsics.data_ptr<scalar_t>(),
        feather ? width.data_ptr<scalar_t>() : nullptr,
        v.data_ptr<scalar_t>(),
        uv0.data_ptr<scalar_t>(),
        uv1.data_ptr<scalar_t>(),
        feather ? opacity.data_ptr<scalar_t>() : nullptr);
    face_kernel<<<blocks(18 * n), kThreads, 0, stream>>>(
        n,
        w,
        owners.data_ptr<int>(),
        core.data_ptr<scalar_t>(),
        depth.data_ptr<scalar_t>(),
        displacement.data_ptr<scalar_t>(),
        intrinsics.data_ptr<scalar_t>(),
        diagonal.numel() ? diagonal.data_ptr<bool>() : nullptr,
        outgoing.data_ptr<int>(),
        incoming.data_ptr<int>(),
        modes.data_ptr<uint8_t>(),
        vmask.data_ptr<int>(),
        vp.data_ptr<int>(),
        fmask.data_ptr<int>(),
        fp.data_ptr<int>(),
        faces.data_ptr<int>(),
        metadata ? cells.data_ptr<int64_t>() : nullptr,
        metadata ? ids.data_ptr<int64_t>() : nullptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

std::vector<torch::Tensor> pixel_grid_mesh_backward(
    const std::vector<torch::Tensor>& s,
    const torch::Tensor& grad_v,
    const torch::Tensor& grad_uv,
    bool grad_intrinsics,
    bool grad_width) {
  const auto &depth = s[0], &displacement = s[1], &k = s[2], &width = s[3], &v = s[4],
             &owners = s[5], &core = s[6], &sizes = s[7], &outgoing = s[8], &incoming = s[9],
             &modes = s[10], &mask = s[11], &prefix = s[12];
  const c10::cuda::CUDAGuard guard(depth.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  int h = depth.size(0), w = depth.size(1), n = h * w, nv = v.size(0);
  bool feather = width.numel() != 0;
  auto gd = torch::zeros_like(depth), gdelta = torch::zeros_like(displacement),
       gk = torch::zeros_like(k), gw = torch::zeros_like(width);
  auto grad = torch::empty_like(v),
       gcore = feather ? torch::zeros_like(core) : torch::empty({0}, depth.options());
  auto gv = grad_v.defined() ? grad_v.contiguous() : torch::Tensor(),
       guv = grad_uv.defined() ? grad_uv.contiguous() : torch::Tensor();
  AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "pixel_grid_mesh_backward", [&] {
    if (nv)
      projection_backward_kernel<<<blocks(nv), kThreads, 0, stream>>>(
          nv,
          h,
          w,
          v.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(),
          gv.defined() ? gv.data_ptr<scalar_t>() : nullptr,
          guv.defined() ? guv.data_ptr<scalar_t>() : nullptr,
          grad.data_ptr<scalar_t>(),
          grad_intrinsics ? gk.data_ptr<scalar_t>() : nullptr);
    if (feather)
      feather_backward_kernel<<<blocks(16 * n), kThreads, 0, stream>>>(
          n,
          owners.data_ptr<int>(),
          core.data_ptr<scalar_t>(),
          outgoing.data_ptr<int>(),
          incoming.data_ptr<int>(),
          modes.data_ptr<uint8_t>(),
          mask.data_ptr<int>(),
          prefix.data_ptr<int>(),
          k.data_ptr<scalar_t>(),
          width.data_ptr<scalar_t>(),
          grad.data_ptr<scalar_t>(),
          gcore.data_ptr<scalar_t>(),
          grad_width ? gw.data_ptr<scalar_t>() : nullptr,
          grad_intrinsics ? gk.data_ptr<scalar_t>() : nullptr);
    corner_backward_kernel<<<blocks(n), kThreads, 0, stream>>>(
        n,
        w,
        depth.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        owners.data_ptr<int>(),
        sizes.data_ptr<uint8_t>(),
        mask.data_ptr<int>(),
        prefix.data_ptr<int>(),
        grad.data_ptr<scalar_t>(),
        feather ? gcore.data_ptr<scalar_t>() : nullptr,
        gd.data_ptr<scalar_t>(),
        gdelta.data_ptr<scalar_t>(),
        grad_intrinsics ? gk.data_ptr<scalar_t>() : nullptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gd, gdelta, gk, gw};
}

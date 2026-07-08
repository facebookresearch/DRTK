#include <c10/util/Half.h>
#include <cstdarg>
#include <memory>
#include "filter2d_kernel.h"

using namespace math;

template <class T, int _up, int _down, int _k_size, int tile_out_w, int tile_out_h>
static __global__ void filter_fused(
    T* x_ptr,
    float* f_ptr,
    T* y_ptr,
    int2 in_size,
    int2 out_size,
    bool reflect,
    bool backward) {
  typedef float scalar_t;

  __shared__ volatile scalar_t sf[_k_size];

  // load filter
  if (threadIdx.x < _k_size) {
    sf[threadIdx.x] = (scalar_t)f_ptr[backward ? threadIdx.x : _k_size - 1 - threadIdx.x];
  }

  constexpr ::detail::StepParameters p_s1 = {
      {tile_out_w, tile_out_h},
      {1, _k_size},
      {1, _up},
      {1, _down},
  };

  constexpr ::detail::StepParameters p_s0 = {
      p_s1.tile_in,
      {_k_size, 1},
      {_up, 1},
      {_down, 1},
  };

  __shared__ volatile scalar_t sx_0[p_s0.tile_in.y][p_s0.tile_in.x];
  __shared__ volatile scalar_t sx_1[p_s1.tile_in.y][p_s1.tile_in.x];

  int2 s1_tile_out_pos = int2({int(blockIdx.x), int(blockIdx.y)}) * p_s1.tile_out;
  int2 s1_tile_in_pos =
      floor_div((s1_tile_out_pos * p_s1.down + p_s1.up - 1 - p_s1.get_pad_0(backward)), p_s1.up);

  int2 s0_tile_out_pos = s1_tile_in_pos;
  int2 s0_tile_in_pos =
      floor_div((s0_tile_out_pos * p_s0.down + p_s0.up - 1 - p_s0.get_pad_0(backward)), p_s0.up);

  int major = (int)blockIdx.z;
  int plane_out_stride = out_size.x * out_size.y;
  int plane_in_stride = in_size.x * in_size.y;

  // load in-tile
  if (reflect) {
    for (int i = (int)threadIdx.x; i < p_s0.tile_in.y * p_s0.tile_in.x; i += (int)blockDim.x) {
      int2 tile_in_xy = ::detail::unpack(i, p_s0.tile_in.x);
      if (all_less(tile_in_xy, p_s0.tile_in)) {
        int2 in_pos = s0_tile_in_pos + tile_in_xy;
        scalar_t v = 0;
        in_pos = max(in_pos, -in_pos);
        in_pos = in_size - max(in_size - 1 - in_pos, -(in_size - 1) + in_pos) - 1;
        v = x_ptr[major * plane_in_stride + in_pos.y * in_size.x + in_pos.x];
        sx_0[tile_in_xy.y][tile_in_xy.x] = v;
      }
    }
  } else {
    for (int i = (int)threadIdx.x; i < p_s0.tile_in.y * p_s0.tile_in.x; i += (int)blockDim.x) {
      int2 tile_in_xy = ::detail::unpack(i, p_s0.tile_in.x);
      if (all_less(tile_in_xy, p_s0.tile_in)) {
        int2 in_pos = s0_tile_in_pos + tile_in_xy;
        scalar_t v = 0;
        if (all_less(in_pos, in_size) && all_greater_or_eq(in_pos, int2({0, 0}))) {
          v = x_ptr[major * plane_in_stride + in_pos.y * in_size.x + in_pos.x];
        }
        sx_0[tile_in_xy.y][tile_in_xy.x] = v;
      }
    }
  }

  // sync after loading data to shared memory
  __syncthreads();

  int2 p_s0_pad_0 = p_s0.get_pad_0(backward);

  for (int idx = (int)threadIdx.x; idx < p_s0.tile_out.x * p_s0.tile_out.y;
       idx += (int)blockDim.x) {
    int2 tile_out_rel = ::detail::unpack(idx, p_s0.tile_out.x);
    int2 out_pos = tile_out_rel + s0_tile_out_pos;
    int2 _in_pos =
        s0_tile_out_pos * p_s0.down + p_s0.up - 1 - p_s0_pad_0 + tile_out_rel * p_s0.down;
    int2 in_pos = floor_div(_in_pos, p_s0.up);
    int2 in_pos_rel = in_pos - s0_tile_in_pos;
    int2 filter_xy = (in_pos + 1) * p_s0.up - _in_pos - 1;

    // If we were not working with tile, we would need to do something like:
    //   `if (all_less(out_pos, out_size))`
    // But we can skip it since we saving result to shared memory and we know
    // that we are not going to go out of bound
    {
      scalar_t v = 0;
#pragma unroll
      for (int y = 0; y < p_s0.k_size.y / p_s0.up.y; y++)
#pragma unroll
        for (int x = 0; x < p_s0.k_size.x / p_s0.up.x; x++)
          v += sx_0[in_pos_rel.y + y][in_pos_rel.x + x] * sf[filter_xy.x + x * p_s0.up.x];
      // If we were not working with tile, we would write result to global memory like:
      //    `y_ptr[out_pos.x + out_pos.y * out_size.x +  major * plane_out_stride] = (T)v;`
      sx_1[tile_out_rel.y][tile_out_rel.x] = v;
    }
  }

  __syncthreads();

  int2 p_s1_pad_0 = p_s1.get_pad_0(backward);

  for (int idx = (int)threadIdx.x; idx < p_s1.tile_out.x * p_s1.tile_out.y;
       idx += (int)blockDim.x) {
    int2 tile_out_rel = ::detail::unpack(idx, p_s1.tile_out.x);
    int2 out_pos = tile_out_rel + s1_tile_out_pos;
    int2 _in_pos =
        s1_tile_out_pos * p_s1.down + p_s1.up - 1 - p_s1_pad_0 + tile_out_rel * p_s1.down;
    int2 in_pos = floor_div(_in_pos, p_s1.up);
    int2 in_pos_rel = in_pos - s1_tile_in_pos;
    int2 filter_xy = (in_pos + 1) * p_s1.up - _in_pos - 1;

    if (all_less(out_pos, out_size)) {
      scalar_t v = 0;
#pragma unroll
      for (int y = 0; y < p_s1.k_size.y / p_s1.up.y; y++)
#pragma unroll
        for (int x = 0; x < p_s1.k_size.x / p_s1.up.x; x++)
          v += sx_1[in_pos_rel.y + y][in_pos_rel.x + x] * sf[filter_xy.y + y * p_s1.up.y];
      y_ptr[out_pos.x + out_pos.y * out_size.x + major * plane_out_stride] = (T)v;
    }
  }
}

template <typename T, int up, int down, int filter, int tile_out_w, int tile_out_h, int tile_limit>
struct GetKernel {
  static void*
  get_kernel_given_tile_limit(int out_size, int& tile_out_w_return, int& tile_out_h_return) {
    if (out_size > tile_limit) {
      constexpr int tile_w = std::min(tile_limit, tile_out_w);
      constexpr int tile_h = std::min(tile_limit, tile_out_h);
      tile_out_w_return = tile_w;
      tile_out_h_return = tile_h;
      return (void*)filter_fused<T, up, down, filter, tile_w, tile_h>;
    } else {
      constexpr int smaller_tile_limit = std::max(4, tile_limit / 2);
      return GetKernel<T, up, down, filter, tile_out_w, tile_out_h, smaller_tile_limit>::
          get_kernel_given_tile_limit(out_size, tile_out_w_return, tile_out_h_return);
    }
  }
};

template <typename T, int up, int down, int filter, int tile_out_w, int tile_out_h>
struct GetKernel<T, up, down, filter, tile_out_w, tile_out_h, 4> {
  static void*
  get_kernel_given_tile_limit(int out_size, int& tile_out_w_return, int& tile_out_h_return) {
    constexpr int tile_limit = 4;
    assert(out_size >= tile_limit);
    constexpr int tile_w = std::min(tile_limit, tile_out_w);
    constexpr int tile_h = std::min(tile_limit, tile_out_h);
    return (void*)filter_fused<T, up, down, filter, tile_w, tile_h>;
  }
};

#define KERNEL_TABLE                                             \
  /* Zero-phase (odd) kernels. */                                \
  /* For the case when there is no upsampling or downsampling */ \
                                                                 \
  CASE(1, 1, 3, 64, 64);                                         \
  CASE(1, 1, 5, 64, 64);                                         \
  CASE(1, 1, 7, 64, 64);                                         \
  CASE(1, 1, 9, 64, 64);                                         \
  CASE(1, 1, 11, 64, 64);                                        \
  CASE(1, 1, 13, 64, 64);                                        \
  CASE(1, 1, 17, 64, 64);                                        \
  CASE(1, 1, 25, 48, 48);                                        \
  CASE(1, 1, 33, 32, 32);                                        \
  CASE(1, 1, 49, 32, 32);                                        \
  CASE(1, 1, 65, 16, 16);                                        \
                                                                 \
  /* Even kernels. For upsampling/downsampling */                \
                                                                 \
  /*  Downsample 2x */                                           \
  CASE(1, 2, 4, 48, 32); /*  2-tap */                            \
  CASE(1, 2, 6, 48, 32); /*  3-tap */                            \
  CASE(1, 2, 8, 48, 32); /*  4-tap */                            \
  CASE(1, 2, 10, 48, 32); /*  5-tap */                           \
  CASE(1, 2, 12, 48, 32); /*  6-tap */                           \
                                                                 \
  /*  Downsample 4x */                                           \
  CASE(1, 4, 16, 24, 16); /*  4-tap */                           \
  CASE(1, 4, 24, 24, 16); /*  6-tap */                           \
                                                                 \
  /*  Downsample 8x */                                           \
  CASE(1, 8, 32, 8, 8); /*  4-tap */                             \
  CASE(1, 8, 48, 8, 8); /*  6-tap */                             \
                                                                 \
  /*  Upsample 2x */                                             \
  CASE(2, 1, 4, 64, 64); /*  2-tap */                            \
  CASE(2, 1, 6, 64, 64); /*  3-tap */                            \
  CASE(2, 1, 8, 64, 64); /*  4-tap */                            \
  CASE(2, 1, 10, 64, 64); /*  5-tap */                           \
  CASE(2, 1, 12, 64, 64); /*  6-tap */                           \
                                                                 \
  /* Upsample 4x */                                              \
  CASE(4, 1, 16, 64, 64); /* 4-tap */                            \
  CASE(4, 1, 24, 64, 64); /* 6-tap */                            \
  /*  Upsample 8x */                                             \
  CASE(8, 1, 32, 64, 64); /*  4-tap */                           \
  CASE(8, 1, 48, 64, 64); /*  6-tap */

#define CASE(UP, DOWN, FILTER, TILE_OUT_W, TILE_OUT_H)                     \
  if (UP == up && DOWN == down && FILTER == filter) {                      \
    return GetKernel<                                                      \
        T,                                                                 \
        UP,                                                                \
        DOWN,                                                              \
        FILTER,                                                            \
        TILE_OUT_W,                                                        \
        TILE_OUT_H,                                                        \
        std::max(TILE_OUT_W, TILE_OUT_H)>::                                \
        get_kernel_given_tile_limit(min_out_size, tile_out_w, tile_out_h); \
  }

template <class T>
void* get_filter_fused_kernel(
    int up,
    int down,
    int filter,
    int min_out_size,
    int& tile_out_w,
    int& tile_out_h) {
  assert(min_out_size >= 4);
  KERNEL_TABLE
  return nullptr;
}

// No implementation for double for now.
template <>
void* get_filter_fused_kernel<double>(
    int up,
    int down,
    int filter,
    int max_out_size,
    int& tile_out_w,
    int& tile_out_h) {
  return nullptr;
}

template void* get_filter_fused_kernel<float>(
    int up,
    int down,
    int filter,
    int min_out_size,
    int& tile_out_w,
    int& tile_out_h);

template void* get_filter_fused_kernel<c10::Half>(
    int up,
    int down,
    int filter,
    int min_out_size,
    int& tile_out_w,
    int& tile_out_h);

#undef CASE
#define CASE(UP, DOWN, FILTER, TILE_OUT_W, TILE_OUT_H) \
  if (UP == up && DOWN == down || all) {               \
    out += string_format(                              \
        "%4d |%4d |%12d |%8d |%8d |%8d\n",             \
        UP,                                            \
        DOWN,                                          \
        FILTER,                                        \
        FILTER / std::max(UP, DOWN),                   \
        TILE_OUT_W,                                    \
        TILE_OUT_H);                                   \
  }

inline std::string string_format(const std::string& fmt_str, ...) {
  int n = ((int)fmt_str.size()) * 2;
  std::unique_ptr<char[]> formatted;
  for (;;) {
    va_list ap;
    va_start(ap, fmt_str);
    formatted = std::unique_ptr<char[]>(new char[n]);
    auto final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
    va_end(ap);
  }
  return formatted.get();
}

std::string print_available_kernels(int up, int down, int filter) {
  bool all = false;
  std::string out;
  std::string header = string_format(
      "%4s |%4s |%12s |%8s |%8s |%8s\n", "up", "down", "kernel size", "n-tap", "tile-w", "tile-h");

  out += string_format(
      "Available kernels for given upscaling/downscaling factor (%1d|%1d):\n", up, down);
  out += header;
  KERNEL_TABLE

  all = true;
  out += string_format("\nAll available kernels:\n");
  out += header;
  KERNEL_TABLE

  return out;
}

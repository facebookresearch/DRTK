#pragma once
#include <cuda_math_helper.h>
#include <string>

namespace detail {
using namespace math;

constexpr inline __host__ __device__ int calc_pad_0(int k_size, int down, int up) {
  if (down == 1 && up == 1)
    return k_size / 2;
  else {
    if (down != 1)
      return (k_size - down + 1) / 2;
    else
      return (k_size + up - 1) / 2;
  }
}

constexpr inline __host__ __device__ int calc_pad_1(int k_size, int down, int up) {
  if (down == 1 && up == 1)
    return (k_size - 1) / 2;
  else {
    if (down != 1)
      return (k_size - down) / 2;
    else
      return (k_size - up) / 2;
  }
}

struct StepParameters {
  constexpr __host__ __device__ StepParameters(int2 tile_out, int2 k_size, int2 up, int2 down)
      : tile_out(tile_out),
        // Element-wise rather than via int2 operators: ROCm declares its vector operator+/-
        // constexpr, but they delegate to non-constexpr operator+=/-=, so they can never be
        // constant-evaluated and StepParameters would stop being usable in constant expressions.
        tile_in(
            {((tile_out.x - 1) * down.x + k_size.x - 1) / up.x + 1,
             ((tile_out.y - 1) * down.y + k_size.y - 1) / up.y + 1}),
        up(up),
        down(down),
        pad_0({calc_pad_0(k_size.x, down.x, up.x), calc_pad_0(k_size.y, down.y, up.y)}),
        pad_1({calc_pad_1(k_size.x, down.x, up.x), calc_pad_1(k_size.y, down.y, up.y)}),
        k_size(k_size) {}
  __host__ __device__ int2 get_pad_0(bool backward) const {
    if (backward)
      return {
          k_size.x - calc_pad_0(k_size.x, up.x, down.x) - 1,
          k_size.y - calc_pad_0(k_size.y, up.y, down.y) - 1};
    else
      return pad_0;
  }
  int2 tile_out;
  int2 tile_in;
  int2 up;
  int2 down;
  int2 pad_0;
  int2 pad_1;
  int2 k_size;
};

constexpr inline __host__ __device__ int2 unpack(unsigned int row_idx, unsigned int stride) {
  return {int(row_idx % stride), int(row_idx / stride)};
}
} // namespace detail

template <typename T>
void* get_filter_fused_kernel(
    int up,
    int down,
    int filter,
    int max_out_size,
    int& tile_out_w,
    int& tile_out_h);

std::string print_available_kernels(int up, int down, int filter);

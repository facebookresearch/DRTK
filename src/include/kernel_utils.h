// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

using at::cuda::detail::getTensorInfo;
using at::cuda::detail::TensorInfo;

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block = CUDA_NUM_THREADS) {
  TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
  constexpr int64_t max_int = std::numeric_limits<int>::max();

  // Round up division for positive number that cannot cause integer overflow
  auto block_num = (N - 1) / max_threads_per_block + 1;
  TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

  return static_cast<int>(block_num);
}

// Dispatch macroses are updated in current pytorch.
// Which causes that the same code is compilable on DGX with the older pytorch
// but no longer compilable on prod
// Thus keeping these macroses here.
#undef AT_PRIVATE_CASE_TYPE
#undef AT_DISPATCH_FLOATING_TYPES
#undef AT_DISPATCH_FLOATING_TYPES_AND_HALF
#undef DISPATCH_FLOAT_AND_HALF

#define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                \
    using scalar_t = type;                         \
    return __VA_ARGS__();                          \
  }

// Dispatches for float and double
#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)       \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

// Dispatches for float, double, and half
#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    switch (_st) {                                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");      \
    }                                                                       \
  }()

// Dispatches for float, double, and half
#define DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, __VA_ARGS__)

// A simple stub to match the dispathcing for multimple types structure, but only for
// for float.
#define DISPATCH_FLOAT(NAME, ...) \
  [&] {                           \
    using scalar_t = float;       \
    return __VA_ARGS__();         \
  }()

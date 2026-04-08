// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <atomic>

// Lock-free atomic primitives for CPU kernels, mirroring CUDA atomics.
//
// On C++20 and later, uses std::atomic_ref which is standards-compliant.
// On C++17, falls back to reinterpret_cast<std::atomic<T>*> with static_assert
// guards on size/alignment (formally UB but works on all major compilers and
// architectures we target: x86-64, aarch64).
//
// TODO: Remove the C++17 fallback once we no longer need to support C++17
// builds (e.g., once all OSS consumers have moved to C++20).

namespace drtk {

namespace detail {

// Helper: get an atomic reference/pointer to a raw address.
// Returns a reference to either a std::atomic_ref or a std::atomic obtained
// via reinterpret_cast.
#if __cplusplus >= 202002L
template <typename T>
inline std::atomic_ref<T> atomic_ref_at(T* addr) {
  return std::atomic_ref<T>(*addr);
}
#else
template <typename T>
inline std::atomic<T>& atomic_ref_at(T* addr) {
  static_assert(
      sizeof(std::atomic<T>) == sizeof(T) && alignof(std::atomic<T>) == alignof(T),
      "atomic<T> must match T layout for reinterpret_cast");
  return *reinterpret_cast<std::atomic<T>*>(addr);
}
#endif

} // namespace detail

// Lock-free atomic addition for float/double gradient accumulation.
// Mirrors CUDA's fastAtomicAdd using a CAS loop.
template <typename scalar_t>
inline void atomic_add(scalar_t* addr, scalar_t val) {
  auto target = detail::atomic_ref_at(addr);
  scalar_t cur = target.load(std::memory_order_relaxed);
  scalar_t desired;
  do {
    desired = cur + val;
  } while (!target.compare_exchange_weak(
      cur, desired, std::memory_order_relaxed, std::memory_order_relaxed));
}

// Lock-free atomic minimum, comparing as unsigned.
// Used for z-buffer packed (depth, triangle_id) values where smaller
// unsigned values represent closer triangles.
template <typename T>
inline void atomic_min_unsigned(T* addr, T val) {
  auto target = detail::atomic_ref_at(addr);
  T cur = target.load(std::memory_order_relaxed);
  using U = std::make_unsigned_t<T>;
  while (static_cast<U>(val) < static_cast<U>(cur)) {
    if (target.compare_exchange_weak(
            cur, val, std::memory_order_relaxed, std::memory_order_relaxed)) {
      break;
    }
  }
}

} // namespace drtk

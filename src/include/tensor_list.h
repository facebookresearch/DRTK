// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

using at::cuda::detail::getTensorInfo;
using at::cuda::detail::TensorInfo;

// TensorInfoCompact is similar to TensorInfo but has fixed number of dims same as
// PackedTensorAccessor. It is supposed to be used on for CUDA `Tensor`s on the host when default
// constructor, assignment and copy constructors are needed, e.g. using in arrays in order to
// transfer them on the device when calling kernels. TensorInfo has a default, assignment and copy
// constructors, but PackedTensorAccessor does not. However TensorInfo is too large to be
// transferred in arrays when calling kernels. On the device, indexing of multidimensional tensors
// produces `TensorAccessor`s. Using RestrictPtrTraits as a default. If aliasing is possible (likely
// to be a very rare case) please use DefaultPtrTraits. Default constructor, assignment and copy
// constructors are only needed on the host aren't available on the device
template <
    typename T,
    typename index_t,
    int N_DIMS,
    template <typename> class PtrTraits = at::RestrictPtrTraits>
struct TensorInfoCompact {
  typedef typename PtrTraits<T>::PtrType PtrType;

  TensorInfoCompact(){};
  __host__ TensorInfoCompact<T, index_t, N_DIMS, PtrTraits>& operator=(
      const TensorInfoCompact<T, index_t, N_DIMS>& other) {
    data = other.data;
    for (int i = 0; i < N_DIMS; ++i) {
      sizes[i] = other.sizes[i];
      strides[i] = other.strides[i];
    }
    return *this;
  };
  __host__ TensorInfoCompact(const TensorInfoCompact<T, index_t, N_DIMS, PtrTraits>& other)
      : data(other.data) {
    for (int i = 0; i < N_DIMS; ++i) {
      sizes[i] = other.sizes[i];
      strides[i] = other.strides[i];
    }
  };
  __host__ TensorInfoCompact(const TensorInfo<T, index_t>& other) : data(other.data) {
    for (int i = 0; i < N_DIMS; ++i) {
      sizes[i] = other.sizes[i];
      strides[i] = other.strides[i];
    }
  }

  __device__ at::TensorAccessor<T, N_DIMS - 1, PtrTraits, index_t> operator[](index_t i) {
    index_t* new_sizes = sizes + 1;
    index_t* new_strides = strides + 1;
    return at::TensorAccessor<T, N_DIMS - 1, PtrTraits, index_t>(
        data + strides[0] * i, new_sizes, new_strides);
  }

  __device__ const at::TensorAccessor<T, N_DIMS - 1, PtrTraits, index_t> operator[](
      index_t i) const {
    const index_t* new_sizes = sizes + 1;
    const index_t* new_strides = strides + 1;
    return at::TensorAccessor<T, N_DIMS - 1, PtrTraits, index_t>(
        data + strides[0] * i, new_sizes, new_strides);
  }

  PtrType data;
  index_t sizes[N_DIMS];
  index_t strides[N_DIMS];
};

template <
    typename scalar_t,
    typename index_t,
    int N_DIMS,
    template <typename> class PtrTraits = at::RestrictPtrTraits>
TensorInfoCompact<scalar_t, index_t, N_DIMS, PtrTraits> getTensorInfoCompact(const at::Tensor& x) {
  auto out = getTensorInfo<scalar_t, index_t>(x);
  assert(out.dims == N_DIMS);
  return out;
}

template <
    typename T,
    typename index_t,
    int N,
    int N_DIMS,
    template <typename> class PtrTraits = at::RestrictPtrTraits>
struct TensorInfoList {
  __device__ __host__ TensorInfoCompact<T, index_t, N_DIMS, PtrTraits>& operator[](int i) {
    return data[i];
  }

  __device__ __host__ const TensorInfoCompact<T, index_t, N_DIMS, PtrTraits>& operator[](
      int i) const {
    return data[i];
  }

  TensorInfoCompact<T, index_t, N_DIMS, PtrTraits> data[N];
};

template <typename IndexType, int N>
struct IndexList {
  __device__ __host__ IndexType& operator[](int i) {
    return data[i];
  }

  __device__ __host__ const IndexType& operator[](int i) const {
    return data[i];
  }

  IndexType data[N] = {0};
};

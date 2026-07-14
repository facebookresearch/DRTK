// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/script.h>

#include <ATen/Parallel.h>
#include <ATen/autocast_mode.h>
#include <c10/util/hash.h>

#ifndef NO_PYBIND
#include <torch/extension.h>
#endif

#include "interpolate_kernel.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <list>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

struct NormalMatrixStructure {
  torch::Tensor crow_indices;
  torch::Tensor col_indices;
  torch::Tensor pair_indices;
};

using NormalMatrixCacheKey = std::tuple<
    int64_t, // target device type
    int64_t, // target device index
    int64_t, // vi device type
    int64_t, // vi device index
    uintptr_t, // storage pointer
    uintptr_t, // data pointer
    int64_t, // size 0
    int64_t, // size 1
    int64_t, // size 2
    int64_t, // stride 0
    int64_t, // stride 1
    int64_t, // stride 2
    int64_t, // storage offset
    int64_t, // dtype
    int64_t, // num_vertices
    uint32_t>; // tensor version

struct NormalMatrixCacheEntry {
  NormalMatrixCacheKey key;
  torch::Tensor owner;
  NormalMatrixStructure structure;
};

using NormalMatrixCacheList = std::list<NormalMatrixCacheEntry>;

constexpr size_t kNormalMatrixStructureCacheMaxSize = 128;

std::mutex& normal_matrix_cache_mutex() {
  static std::mutex cache_mutex;
  return cache_mutex;
}

NormalMatrixCacheList& normal_matrix_cache_lru() {
  static NormalMatrixCacheList cache;
  return cache;
}

std::unordered_map<
    NormalMatrixCacheKey,
    NormalMatrixCacheList::iterator,
    c10::hash<NormalMatrixCacheKey>>&
normal_matrix_cache_index() {
  static std::unordered_map<
      NormalMatrixCacheKey,
      NormalMatrixCacheList::iterator,
      c10::hash<NormalMatrixCacheKey>>
      index;
  return index;
}

NormalMatrixCacheKey normal_matrix_structure_cache_key(
    const torch::Tensor& vi,
    int64_t num_vertices,
    const c10::Device& target_device) {
  // Hashing CUDA tensor contents would synchronize this hot path. PyTorch's
  // version counter is the available no-sync mutation signal, so in-place
  // topology edits naturally miss the cache and rebuild the CSR structure.
  // The public op checks strided layout before cache lookup; strided tensors
  // have storage, so storage/data pointers are safe no-sync identity fields.
  return NormalMatrixCacheKey(
      static_cast<int64_t>(target_device.type()),
      static_cast<int64_t>(target_device.index()),
      static_cast<int64_t>(vi.device().type()),
      static_cast<int64_t>(vi.device().index()),
      reinterpret_cast<uintptr_t>(vi.storage().unsafeGetStorageImpl()),
      reinterpret_cast<uintptr_t>(vi.data_ptr()),
      vi.size(0),
      vi.size(1),
      vi.size(2),
      vi.stride(0),
      vi.stride(1),
      vi.stride(2),
      vi.storage_offset(),
      static_cast<int64_t>(vi.scalar_type()),
      num_vertices,
      vi.unsafeGetTensorImpl()->version_counter().current_version());
}

torch::Tensor normal_matrix_structure_to_device(
    const torch::Tensor& tensor,
    const c10::Device& device) {
  if (tensor.device() == device) {
    return tensor;
  }
  return tensor.to(device, tensor.scalar_type(), /*non_blocking=*/false, /*copy=*/true);
}

NormalMatrixStructure build_normal_matrix_structure(
    const torch::Tensor& vi,
    int64_t num_vertices,
    const c10::Device& target_device) {
  TORCH_CHECK(
      num_vertices >= 0, "interpolation_normal_matrix(): expected num_vertices to be non-negative");
  TORCH_CHECK(
      num_vertices <= std::numeric_limits<int32_t>::max(),
      "interpolation_normal_matrix(): expected num_vertices to fit in int32");

  // Cache misses build topology on CPU. For CUDA vi this device-to-host copy
  // synchronizes, so iterative callers should keep topology identity/version
  // stable and stay on cache hits.
  auto vi_cpu = vi.detach().to(torch::kCPU, torch::kInt32).contiguous();
  const int64_t N = vi_cpu.size(0);
  const int64_t F = vi_cpu.size(1);
  const int64_t num_pairs = N * F * 9;
  TORCH_CHECK(
      num_vertices > 0 || num_pairs == 0,
      "interpolation_normal_matrix(): expected num_vertices to be positive when faces are present");

  std::vector<int64_t> keys(num_pairs);
  const int32_t* vi_ptr = vi_cpu.data_ptr<int32_t>();
  std::atomic<bool> invalid_vertex_index{false};
  at::parallel_for(0, N * F, /*grain_size=*/4096, [&](int64_t begin, int64_t end) {
    for (int64_t face_index = begin; face_index < end; ++face_index) {
      const int32_t* face = vi_ptr + face_index * 3;
      const int64_t row0 = face[0];
      const int64_t row1 = face[1];
      const int64_t row2 = face[2];
      const int64_t rows[3] = {row0, row1, row2};
      const int64_t base = face_index * 9;

      if (row0 < 0 || row0 >= num_vertices || row1 < 0 || row1 >= num_vertices || row2 < 0 ||
          row2 >= num_vertices) {
        invalid_vertex_index.store(true, std::memory_order_relaxed);
        for (int k = 0; k < 9; ++k) {
          keys[base + k] = 0;
        }
        continue;
      }

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          keys[base + i * 3 + j] = rows[i] * num_vertices + rows[j];
        }
      }
    }
  });
  TORCH_CHECK(
      !invalid_vertex_index.load(std::memory_order_relaxed),
      "interpolation_normal_matrix(): vi contains a vertex index outside [0, num_vertices)");

  std::vector<int64_t> unique_keys = keys;
  std::sort(unique_keys.begin(), unique_keys.end());
  unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());
  TORCH_CHECK(
      unique_keys.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "interpolation_normal_matrix(): normal matrix has too many nonzeros for int32 value indices");

  auto long_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
  auto int_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  auto crow_cpu = torch::empty({num_vertices + 1}, long_options);
  auto col_cpu = torch::empty({static_cast<int64_t>(unique_keys.size())}, long_options);
  auto pair_cpu = torch::empty({N, F, 9}, int_options);

  std::vector<int64_t> row_counts(num_vertices);
  int64_t* col_ptr = col_cpu.data_ptr<int64_t>();
  for (size_t value_index = 0; value_index < unique_keys.size(); ++value_index) {
    const int64_t key = unique_keys[value_index];
    const int64_t row = key / num_vertices;
    const int64_t col = key - row * num_vertices;
    row_counts[static_cast<size_t>(row)] += 1;
    col_ptr[value_index] = col;
  }

  int64_t* crow_ptr = crow_cpu.data_ptr<int64_t>();
  crow_ptr[0] = 0;
  for (int64_t row = 0; row < num_vertices; ++row) {
    crow_ptr[row + 1] = crow_ptr[row] + row_counts[static_cast<size_t>(row)];
  }

  int32_t* pair_ptr = pair_cpu.data_ptr<int32_t>();
  at::parallel_for(0, num_pairs, /*grain_size=*/4096, [&](int64_t begin, int64_t end) {
    for (int64_t pair_index = begin; pair_index < end; ++pair_index) {
      const auto found = std::lower_bound(unique_keys.begin(), unique_keys.end(), keys[pair_index]);
      pair_ptr[pair_index] = static_cast<int32_t>(found - unique_keys.begin());
    }
  });

  return {
      normal_matrix_structure_to_device(crow_cpu, target_device),
      normal_matrix_structure_to_device(col_cpu, target_device),
      normal_matrix_structure_to_device(pair_cpu, target_device),
  };
}

NormalMatrixStructure cached_normal_matrix_structure(
    const torch::Tensor& vi,
    int64_t num_vertices,
    const c10::Device& target_device) {
  const NormalMatrixCacheKey key =
      normal_matrix_structure_cache_key(vi, num_vertices, target_device);
  {
    std::lock_guard<std::mutex> lock(normal_matrix_cache_mutex());
    auto& cache = normal_matrix_cache_lru();
    auto& index = normal_matrix_cache_index();
    const auto found = index.find(key);
    if (found != index.end()) {
      cache.splice(cache.begin(), cache, found->second);
      return found->second->structure;
    }
  }

  // Build outside the global cache mutex so unrelated misses can proceed.
  // Matching concurrent misses may duplicate work; the second lookup below
  // prevents duplicate cache insertion.
  NormalMatrixStructure structure = build_normal_matrix_structure(vi, num_vertices, target_device);

  std::lock_guard<std::mutex> lock(normal_matrix_cache_mutex());
  auto& cache = normal_matrix_cache_lru();
  auto& index = normal_matrix_cache_index();
  const auto found = index.find(key);
  if (found != index.end()) {
    cache.splice(cache.begin(), cache, found->second);
    return found->second->structure;
  }

  while (cache.size() >= kNormalMatrixStructureCacheMaxSize) {
    index.erase(cache.back().key);
    cache.pop_back();
  }
  // Keep vi alive while the entry is cached so storage/data pointers cannot be
  // recycled into a stale hit. The LRU bound caps retained topology memory.
  cache.push_front({key, vi, structure});
  index.emplace(cache.front().key, cache.begin());
  return structure;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
interpolation_normal_matrix_forward_with_pairs(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t num_vertices) {
  TORCH_CHECK(
      vi.defined() && index_img.defined() && bary_img.defined(),
      "interpolation_normal_matrix(): expected all inputs to be defined");
  TORCH_CHECK(
      vi.device() == index_img.device() && vi.device() == bary_img.device(),
      "interpolation_normal_matrix(): expected all inputs to be on same device");
  TORCH_CHECK(
      vi.dtype() == torch::kInt32,
      "interpolation_normal_matrix(): expected vi to have int32 type, but vi has ",
      vi.dtype());
  TORCH_CHECK(
      index_img.dtype() == torch::kInt32,
      "interpolation_normal_matrix(): expected index_img to have int32 type, but index_img has ",
      index_img.dtype());
  TORCH_CHECK(
      bary_img.is_floating_point(),
      "interpolation_normal_matrix(): expected bary_img to have floating point type");
  TORCH_CHECK(
      vi.layout() == torch::kStrided && index_img.layout() == torch::kStrided &&
          bary_img.layout() == torch::kStrided,
      "interpolation_normal_matrix(): expected all inputs to have torch.strided layout");
  TORCH_CHECK(
      vi.dim() == 3 && index_img.dim() == 3 && bary_img.dim() == 4 && vi.size(2) == 3 &&
          bary_img.size(1) == 3,
      "interpolation_normal_matrix(): expected vi [N,F,3], index_img [N,H,W], bary_img [N,3,H,W]");
  TORCH_CHECK(
      vi.size(0) == index_img.size(0) && vi.size(0) == bary_img.size(0) &&
          index_img.size(1) == bary_img.size(2) && index_img.size(2) == bary_img.size(3),
      "interpolation_normal_matrix(): expected vi, index_img and bary_img shapes to agree");

  NormalMatrixStructure structure =
      cached_normal_matrix_structure(vi, num_vertices, bary_img.device());
  auto values = bary_img.is_cuda()
      ? interpolation_normal_matrix_values_cuda(
            structure.pair_indices, index_img, bary_img, structure.col_indices.numel())
      : interpolation_normal_matrix_values_cpu(
            structure.pair_indices, index_img, bary_img, structure.col_indices.numel());
  return std::make_tuple(
      structure.crow_indices, structure.col_indices, values, structure.pair_indices);
}

} // namespace

// Dispatch function
torch::Tensor interpolate(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("interpolate_ext::interpolate", "")
                       .typed<decltype(interpolate)>();
  return op.call(vert_attributes, vi, index_img, bary_img);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> interpolation_matrix(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("interpolate_ext::interpolation_matrix", "")
                       .typed<decltype(interpolation_matrix)>();
  return op.call(vi, index_img, bary_img);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolation_normal_matrix(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t num_vertices) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("interpolate_ext::interpolation_normal_matrix", "")
                       .typed<decltype(interpolation_normal_matrix)>();
  return op.call(vi, index_img, bary_img, num_vertices);
}

torch::Tensor interpolation_normal_matrix_values(
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t nnz) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("interpolate_ext::interpolation_normal_matrix_values", "")
                       .typed<decltype(interpolation_normal_matrix_values)>();
  return op.call(pair_indices, index_img, bary_img, nnz);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolation_normal_matrix_cpu(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t num_vertices) {
  const auto out =
      interpolation_normal_matrix_forward_with_pairs(vi, index_img, bary_img, num_vertices);
  return std::make_tuple(std::get<0>(out), std::get<1>(out), std::get<2>(out));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolation_normal_matrix_cuda(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t num_vertices) {
  const auto out =
      interpolation_normal_matrix_forward_with_pairs(vi, index_img, bary_img, num_vertices);
  return std::make_tuple(std::get<0>(out), std::get<1>(out), std::get<2>(out));
}

// Ideally we would need to turn off autograd handling and re-dispatch, but we just call
// kernels directly (CUDA or CPU based on input device)
class InterpolateFunction : public torch::autograd::Function<InterpolateFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& vert_attributes,
      const torch::Tensor& vi,
      const torch::Tensor& index_img,
      const torch::Tensor& bary_img) {
    ctx->set_materialize_grads(false);
    std::vector<torch::Tensor> save_list;
    save_list.push_back(vert_attributes);
    save_list.push_back(vi);
    save_list.push_back(index_img);
    save_list.push_back(bary_img);
    ctx->save_for_backward(save_list);
    auto fwd = vert_attributes.is_cuda()
        ? interpolate_cuda(vert_attributes, vi, index_img, bary_img)
        : interpolate_cpu(vert_attributes, vi, index_img, bary_img);
    return {fwd};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& vert_attributes = saved[0];
    const torch::Tensor& vi = saved[1];
    const torch::Tensor& index_img = saved[2];
    const torch::Tensor& bary_img = saved[3];
    bool bary_img_requires_grad = bary_img.requires_grad();
    bool vert_requires_grad = vert_attributes.requires_grad();

    torch::autograd::tensor_list out;
    if ((!bary_img_requires_grad && !vert_requires_grad) || !grad_outputs[0].defined()) {
      out.resize(4);
      return out;
    }
    auto grad_out = vert_attributes.is_cuda()
        ? interpolate_cuda_backward(grad_outputs[0], vert_attributes, vi, index_img, bary_img)
        : interpolate_cpu_backward(grad_outputs[0], vert_attributes, vi, index_img, bary_img);

    out.push_back(std::get<0>(grad_out));
    out.emplace_back();
    out.emplace_back();
    out.push_back(std::get<1>(grad_out));
    return out;
  }
};

torch::Tensor interpolate_autograd(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  return InterpolateFunction::apply(vert_attributes, vi, index_img, bary_img)[0];
}

class InterpolationMatrixFunction : public torch::autograd::Function<InterpolationMatrixFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& vi,
      const torch::Tensor& index_img,
      const torch::Tensor& bary_img) {
    ctx->set_materialize_grads(false);

    auto fwd = bary_img.is_cuda() ? interpolation_matrix_cuda(vi, index_img, bary_img)
                                  : interpolation_matrix_cpu(vi, index_img, bary_img);
    const torch::Tensor& crow_indices = std::get<0>(fwd);
    const torch::Tensor& col_indices = std::get<1>(fwd);
    const torch::Tensor& values = std::get<2>(fwd);
    const torch::Tensor& row_pixels = std::get<3>(fwd);

    ctx->save_for_backward({vi, index_img, bary_img, row_pixels});
    ctx->mark_non_differentiable({crow_indices, col_indices, row_pixels});
    return {crow_indices, col_indices, values, row_pixels};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& vi = saved[0];
    const torch::Tensor& index_img = saved[1];
    const torch::Tensor& bary_img = saved[2];
    const torch::Tensor& row_pixels = saved[3];

    torch::autograd::tensor_list out(3);
    if (!bary_img.requires_grad() || grad_outputs.size() < 3 || !grad_outputs[2].defined()) {
      return out;
    }

    out[2] = bary_img.is_cuda()
        ? interpolation_matrix_cuda_backward(grad_outputs[2], vi, index_img, bary_img, row_pixels)
        : interpolation_matrix_cpu_backward(grad_outputs[2], vi, index_img, bary_img, row_pixels);
    return out;
  }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
interpolation_matrix_autograd(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  auto out = InterpolationMatrixFunction::apply(vi, index_img, bary_img);
  return std::make_tuple(out[0], out[1], out[2], out[3]);
}

class InterpolationNormalMatrixFunction
    : public torch::autograd::Function<InterpolationNormalMatrixFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& vi,
      const torch::Tensor& index_img,
      const torch::Tensor& bary_img,
      int64_t num_vertices) {
    ctx->set_materialize_grads(false);
    const auto fwd =
        interpolation_normal_matrix_forward_with_pairs(vi, index_img, bary_img, num_vertices);
    const torch::Tensor& crow_indices = std::get<0>(fwd);
    const torch::Tensor& col_indices = std::get<1>(fwd);
    const torch::Tensor& values = std::get<2>(fwd);
    const torch::Tensor& pair_indices = std::get<3>(fwd);

    ctx->save_for_backward({pair_indices, index_img, bary_img});
    ctx->mark_non_differentiable({crow_indices, col_indices});
    return {crow_indices, col_indices, values};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& pair_indices = saved[0];
    const torch::Tensor& index_img = saved[1];
    const torch::Tensor& bary_img = saved[2];

    torch::autograd::tensor_list out(4);
    if (!bary_img.requires_grad() || grad_outputs.size() < 3 || !grad_outputs[2].defined()) {
      return out;
    }

    out[2] = bary_img.is_cuda() ? interpolation_normal_matrix_values_cuda_backward(
                                      grad_outputs[2], pair_indices, index_img, bary_img)
                                : interpolation_normal_matrix_values_cpu_backward(
                                      grad_outputs[2], pair_indices, index_img, bary_img);
    return out;
  }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolation_normal_matrix_autograd(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t num_vertices) {
  auto out = InterpolationNormalMatrixFunction::apply(vi, index_img, bary_img, num_vertices);
  return std::make_tuple(out[0], out[1], out[2]);
}

class InterpolationNormalMatrixValuesFunction
    : public torch::autograd::Function<InterpolationNormalMatrixValuesFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& pair_indices,
      const torch::Tensor& index_img,
      const torch::Tensor& bary_img,
      int64_t nnz) {
    ctx->set_materialize_grads(false);
    ctx->save_for_backward({pair_indices, index_img, bary_img});
    auto values = bary_img.is_cuda()
        ? interpolation_normal_matrix_values_cuda(pair_indices, index_img, bary_img, nnz)
        : interpolation_normal_matrix_values_cpu(pair_indices, index_img, bary_img, nnz);
    return {values};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const torch::Tensor& pair_indices = saved[0];
    const torch::Tensor& index_img = saved[1];
    const torch::Tensor& bary_img = saved[2];

    torch::autograd::tensor_list out(4);
    if (!bary_img.requires_grad() || grad_outputs.empty() || !grad_outputs[0].defined()) {
      return out;
    }

    out[2] = bary_img.is_cuda() ? interpolation_normal_matrix_values_cuda_backward(
                                      grad_outputs[0], pair_indices, index_img, bary_img)
                                : interpolation_normal_matrix_values_cpu_backward(
                                      grad_outputs[0], pair_indices, index_img, bary_img);
    return out;
  }
};

torch::Tensor interpolation_normal_matrix_values_autograd(
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t nnz) {
  return InterpolationNormalMatrixValuesFunction::apply(pair_indices, index_img, bary_img, nnz)[0];
}

torch::Tensor interpolate_autocast(
    const torch::Tensor& vert_attributes,
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return interpolate(
      at::autocast::cached_cast(torch::kFloat32, vert_attributes),
      vi,
      index_img,
      at::autocast::cached_cast(torch::kFloat32, bary_img));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
interpolation_matrix_autocast(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return interpolation_matrix(vi, index_img, at::autocast::cached_cast(torch::kFloat32, bary_img));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolation_normal_matrix_autocast(
    const torch::Tensor& vi,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t num_vertices) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return interpolation_normal_matrix(
      vi, index_img, at::autocast::cached_cast(torch::kFloat32, bary_img), num_vertices);
}

torch::Tensor interpolation_normal_matrix_values_autocast(
    const torch::Tensor& pair_indices,
    const torch::Tensor& index_img,
    const torch::Tensor& bary_img,
    int64_t nnz) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  return interpolation_normal_matrix_values(
      pair_indices, index_img, at::autocast::cached_cast(torch::kFloat32, bary_img), nnz);
}

#ifndef NO_PYBIND
// Just so that we can import this file as a Python module to get the path and
// import the Torch ops.
PYBIND11_MODULE(interpolate_ext, m) {}
#endif

TORCH_LIBRARY(interpolate_ext, m) {
  m.def(
      "interpolate(Tensor vert_attributes, Tensor vi, Tensor index_img, Tensor bary_img) -> Tensor");
  m.def(
      "interpolation_matrix(Tensor vi, Tensor index_img, Tensor bary_img) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "interpolation_normal_matrix(Tensor vi, Tensor index_img, Tensor bary_img, int num_vertices) -> (Tensor, Tensor, Tensor)");
  m.def(
      "interpolation_normal_matrix_values(Tensor pair_indices, Tensor index_img, Tensor bary_img, int nnz) -> Tensor");
}

TORCH_LIBRARY_IMPL(interpolate_ext, Autograd, m) {
  m.impl("interpolate", &interpolate_autograd);
  m.impl("interpolation_matrix", &interpolation_matrix_autograd);
  m.impl("interpolation_normal_matrix", &interpolation_normal_matrix_autograd);
  m.impl("interpolation_normal_matrix_values", &interpolation_normal_matrix_values_autograd);
}

TORCH_LIBRARY_IMPL(interpolate_ext, Autocast, m) {
  m.impl("interpolate", interpolate_autocast);
  m.impl("interpolation_matrix", interpolation_matrix_autocast);
  m.impl("interpolation_normal_matrix", interpolation_normal_matrix_autocast);
  m.impl("interpolation_normal_matrix_values", interpolation_normal_matrix_values_autocast);
}

TORCH_LIBRARY_IMPL(interpolate_ext, CUDA, m) {
  m.impl("interpolate", &interpolate_cuda);
  m.impl("interpolation_matrix", &interpolation_matrix_cuda);
  m.impl("interpolation_normal_matrix", &interpolation_normal_matrix_cuda);
  m.impl("interpolation_normal_matrix_values", &interpolation_normal_matrix_values_cuda);
}

TORCH_LIBRARY_IMPL(interpolate_ext, CPU, m) {
  m.impl("interpolate", &interpolate_cpu);
  m.impl("interpolation_matrix", &interpolation_matrix_cpu);
  m.impl("interpolation_normal_matrix", &interpolation_normal_matrix_cpu);
  m.impl("interpolation_normal_matrix_values", &interpolation_normal_matrix_values_cpu);
}

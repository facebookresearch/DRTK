#include "filter2d_kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

using namespace math;

torch::Tensor
filter2d_fused(torch::Tensor x, torch::Tensor f, int _up, int _down, bool backward, bool reflect) {
  TORCH_CHECK(x.device().is_cuda(), "x is not a CUDA tensor.")
  TORCH_CHECK(x.is_contiguous(), "x is not contiguous.")
  TORCH_CHECK(f.is_contiguous(), "f is not contiguous.")
  TORCH_CHECK(f.device() == x.device(), "f must reside on the same device as x");
  TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
  TORCH_CHECK(x.dim() == 4, "x must be rank 4");
  TORCH_CHECK(f.dim() == 1, "f must be rank 1");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

  int2 in_size = {int(x.size(3)), int(x.size(2))};
  int _k_size = int(f.size(0));

  int size_major = x.size(0) * x.size(1);
  TORCH_CHECK(
      size_major <= 65535, "<number of channels> X <batch size> should be less or equal to 65535");

  TORCH_CHECK(_k_size >= 1, "f must be at least 1x1");
  TORCH_CHECK(_up >= 1, "upsampling factor must be at least 1");
  TORCH_CHECK(_down >= 1, "downsampling factor must be at least 1");

  int min_out_size = -1;
  {
    ::detail::StepParameters p = {
        {-1, -1},
        {_k_size, _k_size},
        {_up, _up},
        {_down, _down},
    };
    int2 out_size = (in_size * p.up + p.pad_0 + p.pad_1 - (p.k_size - 1) + p.down - 1) / p.down;
    min_out_size = std::min(out_size.x, out_size.y);
  }

  int2 tile_out = {-1, -1};
  void* kernel = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filter2d_cuda", [&] {
    kernel = get_filter_fused_kernel<scalar_t>(
        _up, _down, _k_size, min_out_size, tile_out.x, tile_out.y);
  });

#ifdef DEBUG_OUTPUT
  printf("\n-------------------\n");
  printf("DEBUG DATA FOR: [up: %d, down: %d]\n", _up, _down);

  printf("In size: [%d, %d]\n", in_size.x, in_size.y);
  printf("min_out_size: %d\n", min_out_size);
#endif

  std::string msg;
  if (kernel == nullptr) {
    msg = print_available_kernels(_up, _down, _k_size);
  }
  TORCH_CHECK(
      kernel != nullptr,
      "Didn't find suitable kernel for the input parameters:\n\t up: ",
      _up,
      "; down: ",
      _down,
      "; kernel size: ",
      _k_size,
      "\n",
      msg);

  ::detail::StepParameters p_s1 = {
      {tile_out.x, tile_out.y},
      {1, _k_size},
      {1, _up},
      {1, _down},
  };

  ::detail::StepParameters p_s0 = {
      p_s1.tile_in,
      {_k_size, 1},
      {_up, 1},
      {_down, 1},
  };

  int2 s0_out_size =
      (in_size * p_s0.up + p_s0.pad_0 + p_s0.pad_1 - (p_s0.k_size - 1) + p_s0.down - 1) / p_s0.down;
  int2 s1_out_size =
      (s0_out_size * p_s1.up + p_s1.pad_0 + p_s1.pad_1 - (p_s1.k_size - 1) + p_s1.down - 1) /
      p_s1.down;

  TORCH_CHECK(s1_out_size.x >= 1 && s1_out_size.y >= 1, "output must be at least 1x1");
  torch::Tensor y = torch::empty({x.size(0), x.size(1), s1_out_size.y, s1_out_size.x}, x.device());

  dim3 block_size =
      dim3(std::max(p_s0.tile_out.x * p_s0.tile_out.y, p_s1.tile_out.x * p_s1.tile_out.y), 1, 1);

  block_size.x = std::min(int(block_size.x), 512);

  auto x_ptr = x.data_ptr();
  auto f_ptr = f.data_ptr();
  auto y_ptr = y.data_ptr();

  dim3 grid_size = dim3(
      (s1_out_size.x + p_s1.tile_out.x - 1) / p_s1.tile_out.x,
      (s1_out_size.y + p_s1.tile_out.y - 1) / p_s1.tile_out.y,
      size_major);

#ifdef DEBUG_OUTPUT
  auto loop = (int(block_size.x) + 512 - 1) / 512;

  printf("In size: [%d, %d]\n", in_size.x, in_size.y);
  printf("Out size: [%d, %d]\n", s1_out_size.x, s1_out_size.y);
  printf("\n");

  printf("Step 0:\n");
  printf("Tile in: [%d, %d]\n", p_s0.tile_in.x, p_s0.tile_in.y);
  printf("Tile out: [%d, %d]\n", p_s0.tile_out.x, p_s0.tile_out.y);
  printf("\n");

  printf("Step 1:\n");
  printf("Tile in: [%d, %d]\n", p_s1.tile_in.x, p_s1.tile_in.y);
  printf("Tile out: [%d, %d]\n", p_s1.tile_out.x, p_s1.tile_out.y);
  printf("\n");

  printf("Block size: %d\n", block_size.x);
  printf("Loop: %d\n", loop);

  printf("Grid size: [%d, %d, %d]\n", grid_size.x, grid_size.y, grid_size.z);

  int shared_memory_floats =
      p_s0.tile_in.x * p_s0.tile_in.y + p_s1.tile_in.x * p_s1.tile_in.y + _k_size;
  printf(
      "Shared memory per block: %dKB\n",
      (int)(shared_memory_floats * torch::scalarTypeToTypeMeta(x.scalar_type()).itemsize()) / 1024);
#endif

  void* args[] = {&x_ptr, &f_ptr, &y_ptr, &in_size, &s1_out_size, &reflect, &backward};

  AT_CUDA_CHECK(
      cudaLaunchKernel(kernel, grid_size, block_size, args, 0, at::cuda::getCurrentCUDAStream()));

  return y;
}

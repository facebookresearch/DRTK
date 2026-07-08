#include "filter2d_kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional/conv.h>
#include <torch/nn/functional/padding.h>
#include <torch/torch.h>

using namespace math;

namespace {
namespace F = torch::nn::functional;

int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

int64_t output_size(int64_t input_size, int64_t filter_size, int64_t up, int64_t down) {
  const int64_t pad =
      ::detail::calc_pad_0(filter_size, down, up) + ::detail::calc_pad_1(filter_size, down, up);
  return (input_size * up + pad - filter_size + down) / down;
}

int64_t effective_pad_0(int64_t filter_size, int64_t up, int64_t down, bool backward) {
  if (backward) {
    return filter_size - ::detail::calc_pad_0(filter_size, up, down) - 1;
  }
  return ::detail::calc_pad_0(filter_size, down, up);
}

torch::Tensor insert_zeros_cpu(torch::Tensor x, int64_t up) {
  if (up == 1) {
    return x;
  }

  const int64_t batch_size = x.size(0);
  const int64_t num_channels = x.size(1);
  const int64_t in_height = x.size(2);
  const int64_t in_width = x.size(3);

  x = x.reshape({batch_size, num_channels, in_height, 1, in_width, 1});
  x = F::pad(
      x,
      F::PadFuncOptions(std::vector<int64_t>{0, up - 1, 0, 0, 0, up - 1})
          .mode(torch::kConstant)
          .value(0.0));
  return x.reshape({batch_size, num_channels, in_height * up, in_width * up});
}

torch::Tensor pad_for_resampling_cpu(
    torch::Tensor x,
    int64_t up,
    int64_t pad_x0,
    int64_t pad_x1,
    int64_t pad_y0,
    int64_t pad_y1,
    bool reflect) {
  TORCH_CHECK(
      pad_x0 >= 0 && pad_x1 >= 0 && pad_y0 >= 0 && pad_y1 >= 0,
      "filter2d padding must be non-negative; filter length is too small for the sampling factors");

  if (!reflect) {
    x = insert_zeros_cpu(x, up);
    if (pad_x0 != 0 || pad_x1 != 0 || pad_y0 != 0 || pad_y1 != 0) {
      x = F::pad(
          x,
          F::PadFuncOptions(std::vector<int64_t>{pad_x0, pad_x1, pad_y0, pad_y1})
              .mode(torch::kConstant)
              .value(0.0));
    }
    return x;
  }

  const int64_t input_pad_x0 = ceil_div(pad_x0, up);
  const int64_t input_pad_x1 = ceil_div(pad_x1, up);
  const int64_t input_pad_y0 = ceil_div(pad_y0, up);
  const int64_t input_pad_y1 = ceil_div(pad_y1, up);

  if (input_pad_x0 != 0 || input_pad_x1 != 0 || input_pad_y0 != 0 || input_pad_y1 != 0) {
    x = F::pad(
        x,
        F::PadFuncOptions(
            std::vector<int64_t>{input_pad_x0, input_pad_x1, input_pad_y0, input_pad_y1})
            .mode(torch::kReflect));
  }

  x = insert_zeros_cpu(x, up);

  const int64_t crop_x0 = input_pad_x0 * up - pad_x0;
  const int64_t crop_x1 = input_pad_x1 * up - pad_x1;
  const int64_t crop_y0 = input_pad_y0 * up - pad_y0;
  const int64_t crop_y1 = input_pad_y1 * up - pad_y1;
  if (crop_x0 != 0 || crop_x1 != 0 || crop_y0 != 0 || crop_y1 != 0) {
    x = x.slice(2, crop_y0, x.size(2) - crop_y1).slice(3, crop_x0, x.size(3) - crop_x1);
  }
  return x;
}

void validate_filter2d_args(torch::Tensor x, torch::Tensor f, int up, int down) {
  TORCH_CHECK(x.is_contiguous(), "x is not contiguous.")
  TORCH_CHECK(f.is_contiguous(), "f is not contiguous.")
  TORCH_CHECK(f.device() == x.device(), "f must reside on the same device as x");
  TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
  TORCH_CHECK(x.dim() == 4, "x must be rank 4");
  TORCH_CHECK(f.dim() == 1, "f must be rank 1");
  TORCH_CHECK(
      x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kFloat ||
          x.scalar_type() == torch::kDouble,
      "x dtype must be float16, float32, or float64");
  TORCH_CHECK(
      x.size(0) >= 1 && x.size(1) >= 1 && x.size(2) >= 1 && x.size(3) >= 1,
      "x dimensions must be non-empty");
  TORCH_CHECK(f.size(0) >= 1, "f must be at least 1x1");
  TORCH_CHECK(up >= 1, "upsampling factor must be at least 1");
  TORCH_CHECK(down >= 1, "downsampling factor must be at least 1");
}

torch::Tensor
filter2d_cpu(torch::Tensor x, torch::Tensor f, int _up, int _down, bool backward, bool reflect) {
  const int64_t k_size = f.size(0);
  const int64_t out_height = output_size(x.size(2), k_size, _up, _down);
  const int64_t out_width = output_size(x.size(3), k_size, _up, _down);
  TORCH_CHECK(out_height >= 1 && out_width >= 1, "output must be at least 1x1");

  const int64_t pad_x0 = effective_pad_0(k_size, _up, _down, backward);
  const int64_t pad_y0 = pad_x0;
  const int64_t total_pad =
      ::detail::calc_pad_0(k_size, _down, _up) + ::detail::calc_pad_1(k_size, _down, _up);
  const int64_t pad_x1 = total_pad - pad_x0;
  const int64_t pad_y1 = total_pad - pad_y0;

  const auto output_scalar_type = x.scalar_type();
  if (output_scalar_type == torch::kHalf) {
    x = x.to(torch::kFloat);
  }

  x = pad_for_resampling_cpu(x, _up, pad_x0, pad_x1, pad_y0, pad_y1, reflect);

  const int64_t num_channels = x.size(1);
  auto filter = backward ? f : f.flip({0});
  filter = filter.to(x.scalar_type()).contiguous();

  auto filter_x = filter.reshape({1, 1, 1, k_size}).repeat({num_channels, 1, 1, 1});
  x = F::conv2d(
      x,
      filter_x,
      F::Conv2dFuncOptions().stride(std::vector<int64_t>{1, _down}).groups(num_channels));

  auto filter_y = filter.reshape({1, 1, k_size, 1}).repeat({num_channels, 1, 1, 1});
  x = F::conv2d(
      x,
      filter_y,
      F::Conv2dFuncOptions().stride(std::vector<int64_t>{_down, 1}).groups(num_channels));

  if (x.scalar_type() != output_scalar_type) {
    x = x.to(output_scalar_type);
  }
  return x.contiguous();
}

torch::Tensor
filter2d_cuda(torch::Tensor x, torch::Tensor f, int _up, int _down, bool backward, bool reflect) {
  TORCH_CHECK(x.device().is_cuda(), "x is not a CUDA tensor.")

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
  torch::Tensor y = torch::empty({x.size(0), x.size(1), s1_out_size.y, s1_out_size.x}, x.options());

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

} // namespace

torch::Tensor
filter2d_fused(torch::Tensor x, torch::Tensor f, int up, int down, bool backward, bool reflect) {
  validate_filter2d_args(x, f, up, down);
  if (x.device().is_cuda()) {
    return filter2d_cuda(x, f, up, down, backward, reflect);
  }
  if (x.device().is_cpu()) {
    return filter2d_cpu(x, f, up, down, backward, reflect);
  }
  TORCH_CHECK(false, "x must be a CPU or CUDA tensor");
}

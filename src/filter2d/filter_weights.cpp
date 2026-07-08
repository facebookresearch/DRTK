#include "filter_weights.h"
#include <c10/util/hash.h>
#include <torch/torch.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <mutex>
#include <tuple>
#include <unordered_map>

template <typename T>
inline T sinc(const T x) {
  static T const taylor_0_bound = std::numeric_limits<T>::epsilon();
  static T const taylor_2_bound = std::sqrt(taylor_0_bound);
  static T const taylor_n_bound = std::sqrt(taylor_2_bound);

  T result = static_cast<T>(1);

  T xpi = x * M_PI;

  if (std::abs(xpi) >= taylor_n_bound) {
    result = std::sin(xpi) / xpi;
  } else {
    if (std::abs(xpi) >= taylor_0_bound) {
      T x2 = xpi * xpi;

      result -= x2 / static_cast<T>(6);

      if (std::abs(xpi) >= taylor_2_bound) {
        result += (x2 * x2) / static_cast<T>(120);
      }
    }
  }
  return result;
}

torch::Tensor make_filter_kernel_kaiser(int n, double fh_s, double fc_s, int m, double gain) {
  n = n * m;
  double L = double(n - 1) / double(m);
  double df = (2.0 * fh_s) / (double(m) / 2.0);
  double A = 2.285 * (n - 1) * M_PI * df + 7.95;

  double beta = 0.0;
  if (A > 50.0) {
    beta = (.1102 * (A - 8.7));
  } else {
    if (A < 21.0)
      beta = 0;
    else {
      beta = 0.5842 * pow(A - 21, 0.4) + 0.07886 * (A - 21);
    }
  }
  auto w =
      torch::empty({n}, torch::TensorOptions().dtype<float>().device(torch::Device::Type::CPU));
  auto* __restrict ptr = w.data_ptr<float>();

  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    double x = double(i);
    x = (x - double(n - 1) / 2.0) / double(m);
    double wi = std::cyl_bessel_i(0, beta * std::pow(1.0 - std::pow(2 * x / L, 2.0), 0.5)) /
        std::cyl_bessel_i(0, beta);
    double v = wi * 2.0 * fc_s * sinc(2.0 * fc_s * x);
    sum += v;
    ptr[i] = v;
  }
  double mul = gain / sum;
  for (int i = 0; i < n; ++i) {
    ptr[i] *= mul;
  }
  return w;
}

torch::Tensor make_filter_kernel_lanczos(int n, double fc_s, int m, double gain) {
  n = n * m;
  auto w =
      torch::empty({n}, torch::TensorOptions().dtype<float>().device(torch::Device::Type::CPU));
  auto* __restrict ptr = w.data_ptr<float>();

  float a = std::ceil(2. * fc_s * (double(n) - 1.) / 2. / double(m));

  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    double x = double(i);
    x = (x - double(n - 1) / 2.0) / double(m);
    double v = 2.0 * fc_s * sinc(2.0 * fc_s * x) * sinc(2.0 * fc_s * x / a) *
        double(std::abs(2 * fc_s * x) < a);
    sum += v;
    ptr[i] = v;
  }
  double mul = gain / sum;
  for (int i = 0; i < n; ++i) {
    ptr[i] *= mul;
  }
  return w;
}

torch::Tensor make_resampling_kernel_kaiser(
    int64_t n,
    double fh_s,
    double fc_s,
    int64_t m,
    double gain,
    torch::Device device) {
  typedef std::tuple<float, float, float, int16_t, int16_t, torch::Device> kernel_key;

  static std::unordered_map<kernel_key, torch::Tensor, c10::hash<kernel_key>> map;
  static std::mutex map_mutex;
  auto key = kernel_key(gain, fh_s, fc_s, n, m, device);
  {
    std::lock_guard<std::mutex> lock(map_mutex);
    auto it = map.find(key);
    if (it != map.end()) {
      return it->second;
    }
  }

  auto kernel = make_filter_kernel_kaiser(n, fh_s, fc_s, m, gain).to(device);
  {
    std::lock_guard<std::mutex> lock(map_mutex);
    auto result = map.emplace(key, kernel);
    return result.first->second;
  }
}

torch::Tensor make_resampling_kernel_lanczos(
    int64_t n,
    double fh_s,
    double fc_s,
    int64_t m,
    double gain,
    torch::Device device) {
  typedef std::tuple<float, float, int16_t, int16_t, torch::Device> kernel_key;

  static std::unordered_map<kernel_key, torch::Tensor, c10::hash<kernel_key>> map;
  static std::mutex map_mutex;
  auto key = kernel_key(gain, fc_s, n, m, device);
  {
    std::lock_guard<std::mutex> lock(map_mutex);
    auto it = map.find(key);
    if (it != map.end()) {
      return it->second;
    }
  }

  auto kernel = make_filter_kernel_lanczos(n, fc_s, m, gain).to(device);
  {
    std::lock_guard<std::mutex> lock(map_mutex);
    auto result = map.emplace(key, kernel);
    return result.first->second;
  }
}

torch::Tensor make_resampling_kernel(
    int64_t n,
    int64_t m,
    double freq_div,
    double gain,
    double alias_guard_band,
    int64_t filter_type,
    torch::Device device) {
  double fh_s = (std::exp2f(0.5f) - 1) / 2 / freq_div;
  double fc_s = 1.0 / 2.0 / freq_div - fh_s * alias_guard_band;

  if (FilterType(filter_type) == FilterType::Kaiser)
    return make_resampling_kernel_kaiser(n, fh_s, fc_s, m, gain, device);
  if (FilterType(filter_type) == FilterType::Lanczos)
    return make_resampling_kernel_lanczos(n, fh_s, fc_s, m, gain, device);
  throw std::runtime_error("Could not find requested filter type");
}

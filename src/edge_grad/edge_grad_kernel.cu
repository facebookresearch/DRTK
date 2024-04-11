#include <c10/cuda/CUDAStream.h>
#include <cassert>

#include "../include/common.h"
#include "../render/helper_math.h"

inline __device__ __host__ float epsclamp(float v, float eps = 1e-8f) {
  return (v < 0) ? min(v, -eps) : max(v, eps);
}

struct TriInfo {
  const float3 p0;
  const float3 normal;
  const float3 v10;
  const float3 v02;
};

__device__ bool pix_in_tri(const TriInfo& tri, const size_t x, const size_t y) {
  const float2 vp0p = make_float2(tri.p0.x - x, tri.p0.y - y) / tri.normal.z;
  const float bary_1 = tri.v02.x * -vp0p.y + tri.v02.y * vp0p.x;
  const float bary_2 = tri.v10.x * -vp0p.y + tri.v10.y * vp0p.x;
  return (bary_1 > 0) && (bary_2 > 0) && ((bary_1 + bary_2) < 1.0);
}

__device__ TriInfo get_tri_info(const float3* v_pix, int n, int V, int3 vipt) {
  const float3 p0 = v_pix[n * V + vipt.x];
  const float3 p1 = v_pix[n * V + vipt.y];
  const float3 p2 = v_pix[n * V + vipt.z];

  const float3 v10 = p1 - p0;
  const float3 v02 = p0 - p2;
  const float3 normal = cross(v02, v10);
  return {p0, normal, v10, v02};
}

__device__ float2 get_dp_db(const float2& v1, const float2& v2) {
  /*
      Computes derivative of the point position with respect to edge displacement
      Args:
          v1: Projection of the normal of the movable triangle onto the plane of consideration (XZ
     or YZ) N x 3 x H x W v2:  Projection of the normal of the fixed triangle onto the plane of
     consideration (XZ or YZ) N x 3 x H x W This function considers partial derivative in a XZ or YZ
     plane. For shortness, only XZ plane will be mentioned, but the same applies for YZ too. We
     consider movement of edge position along X and along Y separately, thus computing partial
     derivatives. Notation: p - point position on the movable triangle in XZ plane for which we
     compute derivatives b - edge displacement vector in XZ plane. Displacement of the edge due to
     displacement of the movable triangle v1 - direction of the displacement vector of the movable
     triangle, also coinsides with it's normal v2 - normal of the fixed triangle v1' - normalized
     projection of v1 onto XZ plane v2' - normalized projection of v2 onto XZ plane t - coordinate
     on the displacement vector, e.i. distance that movable triangle was moved along v1 So: p = p_0
     + v1' * t Since one triangle is fixed, and the other moves, edge displacement b will bi in
     plane of the fixed triangle We can compute direction of b by rotating v2' by 90 deg. Returns:
     dp/db_x - derivative of the point position (p) with respect to edge displacement along X axis
     (bx) Note: We consider only parallel displacement of the movable triangle, so p moves along v1
            We do not need to consider rotation, as rotation in XZ plane around the point of
     intersection won't move the point of intersection Derivative is computed as: dp/db_x = dp/dt *
     dt/db_x Where: dt_db_x = 1 / b_x Edge displacement vector b is computed as: b_dir = rotate(v2,
     90deg) b = b_dir / dot(b_dir, v1') Thus: dt/db_x = dot(b_dir, v1') / b_dir_x The dp/dt part is
     simply not normalized projection of v1 onto XZ plane: dp/dt = v1_xz
  */
  const auto v1_n = normalize(v1);
  const auto v2_n = normalize(v2);
  const float2 b = {v2_n.y, -v2_n.x};
  const float b_dot_v1 = dot(b, v1_n);
  const float dv1_dbx = b_dot_v1 / epsclamp(b.x);
  return dv1_dbx * v1;
}

__global__ void edge_grad_bwd_kernel(
    int N,
    int C,
    int H,
    int W,
    int V,
    const float3* v_pix,
    const float* img,
    const int32_t* index_img,
    const int3* vi,
    const float* grad_output,
    float* grad_v_pix_img) {
  const int count = N * H * W;
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < count; id += blockDim.x * gridDim.x) {
    const int x = id % W;
    const int y = (id / W) % H;
    const int n = id / (H * W);

    if (x > 0 && y > 0 && x < (W - 1) && y < (H - 1)) {
      const int32_t center_index = index_img[n * H * W + y * W + x];
      const int32_t right_index = index_img[n * H * W + y * W + (x + 1)];
      const int32_t down_index = index_img[n * H * W + (y + 1) * W + x];

      const bool c_valid = (center_index >= 0);
      const bool r_valid = (right_index >= 0);
      const bool d_valid = (down_index >= 0);
      const bool lr_diff = (center_index != right_index);
      const bool ud_diff = (center_index != down_index);

      const bool x_both_valid = c_valid && r_valid;
      const bool y_both_valid = c_valid && d_valid;

      const int3 bad_vipt = {0, 0, 0};
      const int3 vipt_center = c_valid ? vi[center_index] : bad_vipt;
      const int3 vipt_right = r_valid ? vi[right_index] : bad_vipt;
      const int3 vipt_down = d_valid ? vi[down_index] : bad_vipt;

      const TriInfo tri_center = get_tri_info(v_pix, n, V, vipt_center);
      const TriInfo tri_right = get_tri_info(v_pix, n, V, vipt_right);
      const TriInfo tri_down = get_tri_info(v_pix, n, V, vipt_down);

      const bool center_pix_in_right_tri = lr_diff && x_both_valid && pix_in_tri(tri_right, x, y);
      const bool right_pix_in_center_tri =
          lr_diff && x_both_valid && pix_in_tri(tri_center, x + 1, y);
      const bool center_pix_in_down_tri = ud_diff && y_both_valid && pix_in_tri(tri_down, x, y);
      const bool down_pix_in_center_tri =
          ud_diff && y_both_valid && pix_in_tri(tri_center, x, y + 1);

      const bool horiz_int = center_pix_in_right_tri && right_pix_in_center_tri;
      const bool vert_int = center_pix_in_down_tri && down_pix_in_center_tri;

      const bool l_over_r = center_pix_in_right_tri && (!right_pix_in_center_tri);
      const bool r_over_l = right_pix_in_center_tri && (!center_pix_in_right_tri);
      const bool u_over_d = center_pix_in_down_tri && (!down_pix_in_center_tri);
      const bool d_over_u = down_pix_in_center_tri && (!center_pix_in_down_tri);

      float grad_dot_x = 0.f;
      float grad_dot_y = 0.f;
      if (lr_diff) {
        for (size_t c = 0; c < C; ++c) {
          grad_dot_x += (img[n * C * H * W + c * H * W + y * W + (x + 1)] -
                         img[n * C * H * W + c * H * W + y * W + x]) *
              (0.5f *
               (grad_output[n * C * H * W + c * H * W + y * W + (x + 1)] +
                grad_output[n * C * H * W + c * H * W + y * W + x]));
        }
      }
      if (ud_diff) {
        for (size_t c = 0; c < C; ++c) {
          grad_dot_y += (img[n * C * H * W + c * H * W + (y + 1) * W + x] -
                         img[n * C * H * W + c * H * W + y * W + x]) *
              (0.5f *
               (grad_output[n * C * H * W + c * H * W + (y + 1) * W + x] +
                grad_output[n * C * H * W + c * H * W + y * W + x]));
        }
      }

      float3 grad_v_pix_center = {0.f, 0.f, 0.f};
      float3 grad_v_pix_right = {0.f, 0.f, 0.f};
      float3 grad_v_pix_down = {0.f, 0.f, 0.f};

      const float3 center_normal = normalize(tri_center.normal);
      const float3 right_normal = normalize(tri_right.normal);
      const float3 down_normal = normalize(tri_down.normal);

      if (!horiz_int) {
        grad_v_pix_center.x += (!c_valid || r_over_l) ? 0.f : grad_dot_x;
        grad_v_pix_right.x += (!r_valid || l_over_r) ? 0.f : grad_dot_x;
      } else {
        // Center triangle moves, right fixed.
        float2 dp_dbx = get_dp_db(
            make_float2(center_normal.x, center_normal.z),
            make_float2(-right_normal.x, -right_normal.z));
        grad_v_pix_center.x += grad_dot_x * dp_dbx.x;
        grad_v_pix_center.z += grad_dot_x * dp_dbx.y;

        // Center triangle fixed, right moves.
        dp_dbx = get_dp_db(
            make_float2(right_normal.x, right_normal.z),
            make_float2(center_normal.x, center_normal.z));
        grad_v_pix_right.x += grad_dot_x * dp_dbx.x;
        grad_v_pix_right.z += grad_dot_x * dp_dbx.y;
      }

      if (!vert_int) {
        grad_v_pix_center.y += (!c_valid || d_over_u) ? 0.f : grad_dot_y;
        grad_v_pix_down.y += (!d_valid || u_over_d) ? 0.f : grad_dot_y;
      } else {
        // Center triangle moves, lower fixed.
        float2 dp_dby = get_dp_db(
            make_float2(center_normal.y, center_normal.z),
            make_float2(-down_normal.y, -down_normal.z));
        grad_v_pix_center.y += grad_dot_y * dp_dby.x;
        grad_v_pix_center.z += grad_dot_y * dp_dby.y;

        // Center triangle fixed, lower moves.
        dp_dby = get_dp_db(
            make_float2(down_normal.y, down_normal.z),
            make_float2(center_normal.y, center_normal.z));
        grad_v_pix_down.y += grad_dot_y * dp_dby.x;
        grad_v_pix_down.z += grad_dot_y * dp_dby.y;
      }

      atomicAdd(grad_v_pix_img + n * H * W * 3 + y * W * 3 + x * 3 + 0, -grad_v_pix_center.x);
      atomicAdd(grad_v_pix_img + n * H * W * 3 + y * W * 3 + x * 3 + 1, -grad_v_pix_center.y);
      atomicAdd(grad_v_pix_img + n * H * W * 3 + y * W * 3 + x * 3 + 2, -grad_v_pix_center.z);

      atomicAdd(grad_v_pix_img + n * H * W * 3 + y * W * 3 + (x + 1) * 3 + 0, -grad_v_pix_right.x);
      atomicAdd(grad_v_pix_img + n * H * W * 3 + y * W * 3 + (x + 1) * 3 + 1, -grad_v_pix_right.y);
      atomicAdd(grad_v_pix_img + n * H * W * 3 + y * W * 3 + (x + 1) * 3 + 2, -grad_v_pix_right.z);

      atomicAdd(grad_v_pix_img + n * H * W * 3 + (y + 1) * W * 3 + x * 3 + 0, -grad_v_pix_down.x);
      atomicAdd(grad_v_pix_img + n * H * W * 3 + (y + 1) * W * 3 + x * 3 + 1, -grad_v_pix_down.y);
      atomicAdd(grad_v_pix_img + n * H * W * 3 + (y + 1) * W * 3 + x * 3 + 2, -grad_v_pix_down.z);
    }
  }
}

class EdgeGradEstimatorFunction : public torch::autograd::Function<EdgeGradEstimatorFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor v_pix,
      const torch::Tensor v_pix_img,
      const torch::Tensor vi,
      const torch::Tensor img,
      const torch::Tensor index_img) {
    ctx->set_materialize_grads(false);
    ctx->save_for_backward({v_pix, img, index_img, vi});
    ctx->saved_data["v_pix_img_requires_grad"] = v_pix_img.requires_grad();
    return {img};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const auto v_pix = saved[0];
    const auto img = saved[1];
    const auto index_img = saved[2];
    const auto vi = saved[3];

    const auto N = img.sizes()[0];
    const auto C = img.sizes()[1];
    const auto H = img.sizes()[2];
    const auto W = img.sizes()[3];
    const auto V = v_pix.sizes()[1];

    // If v_pix_img doesn't require grad, we don't need to do anything.
    if (!ctx->saved_data["v_pix_img_requires_grad"].toBool()) {
      return {torch::Tensor(), torch::Tensor(), torch::Tensor(), grad_outputs[0], torch::Tensor()};
    }

    auto grad_v_pix_img = torch::zeros(
        {N, H, W, 3}, torch::TensorOptions().dtype(v_pix.dtype()).device(v_pix.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const auto count = N * H * W;
    const auto nthreads = 256;
    const auto nblocks = (count + nthreads - 1) / nthreads;
    edge_grad_bwd_kernel<<<nblocks, nthreads, 0, stream>>>(
        N,
        C,
        H,
        W,
        V,
        reinterpret_cast<float3*>(v_pix.data_ptr<float>()),
        img.data_ptr<float>(),
        index_img.data_ptr<int>(),
        reinterpret_cast<int3*>(vi.data_ptr<int>()),
        grad_outputs[0].data_ptr<float>(),
        grad_v_pix_img[0].data_ptr<float>());

    return {torch::Tensor(), grad_v_pix_img, torch::Tensor(), grad_outputs[0], torch::Tensor()};
  }
};

torch::Tensor edge_grad_estimator_autograd(
    const torch::Tensor v_pix,
    const torch::Tensor v_pix_img,
    const torch::Tensor vi,
    const torch::Tensor img,
    const torch::Tensor index_img) {
  CHECK_INPUT(v_pix);
  CHECK_INPUT(v_pix_img);
  CHECK_INPUT(vi);
  CHECK_INPUT(img);
  CHECK_INPUT(index_img);
  CHECK_3DIMS(v_pix);
  CHECK_3DIMS(index_img);

  TORCH_CHECK(vi.dtype() == torch::kInt32, "Vertex indices must have int32 dtype.");
  TORCH_CHECK(index_img.dtype() == torch::kInt32, "Index image must have int32 dtype.");

  return EdgeGradEstimatorFunction::apply(v_pix, v_pix_img, vi, img, index_img)[0];
}

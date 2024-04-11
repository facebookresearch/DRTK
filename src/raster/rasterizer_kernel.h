#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void packed_rasterize_cuda(
    int n,
    int w,
    int h,
    int nprims,
    int nverts,
    const float3* verts,
    const int3* vind,
    float* depth_img,
    int* index_img,
    int* packedindex_img,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

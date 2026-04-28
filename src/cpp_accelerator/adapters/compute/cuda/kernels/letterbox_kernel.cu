#include <cuda_runtime.h>
#include <cstdint>
#include "src/cpp_accelerator/adapters/compute/cuda/kernels/letterbox_kernel.h"

namespace jrb::infrastructure::cuda {

__global__ void letterbox_resize_kernel(const uint8_t* src, int src_w, int src_h, int src_c,
                                        float* dst, int dst_w, int dst_h, float scale, int pad_x,
                                        int pad_y) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst_w || y >= dst_h) {
    return;
  }

  int src_x = static_cast<int>((x - pad_x) / scale);
  int src_y = static_cast<int>((y - pad_y) / scale);

  bool in_bounds = src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h;

  for (int c = 0; c < 3; ++c) {
    float value = 0.0f;
    if (in_bounds) {
      // Preserve channel mapping from original Preprocess():
      // src_c==4 (BGRA-like): c=0->src[1], c=1->src[2], c=2->src[2]
      // src_c==3: c->src[c]
      int actual_c = (src_c == 4) ? (c == 2 ? 2 : c + 1) : c;
      value = static_cast<float>(src[(src_y * src_w + src_x) * src_c + actual_c]) / 255.0f;
    }
    dst[c * dst_w * dst_h + y * dst_w + x] = value;
  }
}

}  // namespace jrb::infrastructure::cuda

// Variant that writes the letterbox result directly into a pre-allocated device buffer.
// Use this when the caller needs GPU memory (e.g. for TensorRT setTensorAddress).
extern "C" cudaError_t cuda_letterbox_resize_to_device(const uint8_t* src_host, int src_w,
                                                       int src_h, int src_c, float* dst_device,
                                                       int dst_w, int dst_h) {
  float scale_x = static_cast<float>(dst_w) / src_w;
  float scale_y = static_cast<float>(dst_h) / src_h;
  float scale = scale_x < scale_y ? scale_x : scale_y;

  int pad_x = (dst_w - static_cast<int>(src_w * scale)) / 2;
  int pad_y = (dst_h - static_cast<int>(src_h * scale)) / 2;

  size_t src_size = static_cast<size_t>(src_w) * src_h * src_c;

  uint8_t* d_src = nullptr;
  cudaError_t err = cudaMalloc(&d_src, src_size);
  if (err != cudaSuccess)
    return err;

  err = cudaMemcpy(d_src, src_host, src_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_src);
    return err;
  }

  dim3 block_size(16, 16);
  dim3 grid_size((dst_w + block_size.x - 1) / block_size.x,
                 (dst_h + block_size.y - 1) / block_size.y);

  jrb::infrastructure::cuda::letterbox_resize_kernel<<<grid_size, block_size>>>(
      d_src, src_w, src_h, src_c, dst_device, dst_w, dst_h, scale, pad_x, pad_y);

  err = cudaDeviceSynchronize();
  cudaFree(d_src);
  if (err != cudaSuccess)
    return err;

  return cudaGetLastError();
}

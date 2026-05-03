#include <cuda_runtime.h>
#include <cstdint>

#include "src/cpp_accelerator/adapters/compute/cuda/kernels/nv12_utils_kernel.h"

// BT.601 limited-range YCbCr → RGB coefficients (matches libavcodec defaults).
// Y: [16..235]  Cb,Cr: [16..240]  → R,G,B: [0..255]
__device__ __forceinline__ void nv12_pixel_to_rgba(int y_raw, int u_raw, int v_raw, uint8_t* r,
                                                   uint8_t* g, uint8_t* b) {
  const int y = y_raw - 16;
  const int u = u_raw - 128;
  const int v = v_raw - 128;

  const int r_val = (298 * y + 409 * v + 128) >> 8;
  const int g_val = (298 * y - 100 * u - 208 * v + 128) >> 8;
  const int b_val = (298 * y + 516 * u + 128) >> 8;

  *r = static_cast<uint8_t>(max(0, min(r_val, 255)));
  *g = static_cast<uint8_t>(max(0, min(g_val, 255)));
  *b = static_cast<uint8_t>(max(0, min(b_val, 255)));
}

// ──────────────────────────────────────────────────────────────────────────────
// NV12 → RGBA
// ──────────────────────────────────────────────────────────────────────────────

__global__ void nv12_to_rgba_kernel(const uint8_t* y_dev, const uint8_t* uv_dev, int pitch,
                                    uint8_t* rgba_dev, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int y_raw  = y_dev[y * pitch + x];
  const int uv_row = y / 2;
  const int uv_col = (x / 2) * 2;
  const int u_raw  = uv_dev[uv_row * pitch + uv_col];
  const int v_raw  = uv_dev[uv_row * pitch + uv_col + 1];

  uint8_t r, g, b;
  nv12_pixel_to_rgba(y_raw, u_raw, v_raw, &r, &g, &b);

  const int out_idx     = (y * width + x) * 4;
  rgba_dev[out_idx]     = r;
  rgba_dev[out_idx + 1] = g;
  rgba_dev[out_idx + 2] = b;
  rgba_dev[out_idx + 3] = 255u;
}

extern "C" cudaError_t cuda_nv12_to_rgba_device(const uint8_t* y_dev, const uint8_t* uv_dev,
                                                 int pitch, uint8_t* rgba_dev, int width,
                                                 int height) {
  const dim3 block(16, 16);
  const dim3 grid((width + 15) / 16, (height + 15) / 16);
  nv12_to_rgba_kernel<<<grid, block>>>(y_dev, uv_dev, pitch, rgba_dev, width, height);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  return cudaGetLastError();
}

// ──────────────────────────────────────────────────────────────────────────────
// RGBA → NV12
// ──────────────────────────────────────────────────────────────────────────────

// BT.601 limited-range RGB → YCbCr.
__device__ __forceinline__ void rgba_pixel_to_nv12(uint8_t r, uint8_t g, uint8_t b, int* y_out,
                                                   int* u_out, int* v_out) {
  *y_out = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
  *u_out = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
  *v_out = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
}

// Y plane — one thread per luma sample.
__global__ void rgba_to_nv12_y_kernel(const uint8_t* rgba_dev, uint8_t* y_dev, int pitch,
                                      int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int src   = (y * width + x) * 4;
  int y_val, u_dummy, v_dummy;
  rgba_pixel_to_nv12(rgba_dev[src], rgba_dev[src + 1], rgba_dev[src + 2], &y_val, &u_dummy,
                     &v_dummy);
  y_dev[y * pitch + x] = static_cast<uint8_t>(max(0, min(y_val, 255)));
}

// UV plane — one thread per 2×2 chroma block (average the four luma neighbours).
__global__ void rgba_to_nv12_uv_kernel(const uint8_t* rgba_dev, uint8_t* uv_dev, int pitch,
                                       int width, int height) {
  const int cx = blockIdx.x * blockDim.x + threadIdx.x;  // chroma column
  const int cy = blockIdx.y * blockDim.y + threadIdx.y;  // chroma row
  const int lx = cx * 2;
  const int ly = cy * 2;
  if (lx >= width || ly >= height) return;

  int u_sum = 0, v_sum = 0, count = 0;
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const int px = lx + dx;
      const int py = ly + dy;
      if (px < width && py < height) {
        const int src = (py * width + px) * 4;
        int y_dummy, u_val, v_val;
        rgba_pixel_to_nv12(rgba_dev[src], rgba_dev[src + 1], rgba_dev[src + 2], &y_dummy, &u_val,
                           &v_val);
        u_sum += u_val;
        v_sum += v_val;
        ++count;
      }
    }
  }

  if (count > 0) {
    const int uv_col         = cx * 2;
    uv_dev[cy * pitch + uv_col]     = static_cast<uint8_t>(max(0, min(u_sum / count, 255)));
    uv_dev[cy * pitch + uv_col + 1] = static_cast<uint8_t>(max(0, min(v_sum / count, 255)));
  }
}

extern "C" cudaError_t cuda_rgba_to_nv12_device(const uint8_t* rgba_dev, uint8_t* y_dev,
                                                 uint8_t* uv_dev, int pitch, int width,
                                                 int height) {
  // Y plane
  {
    const dim3 block(16, 16);
    const dim3 grid((width + 15) / 16, (height + 15) / 16);
    rgba_to_nv12_y_kernel<<<grid, block>>>(rgba_dev, y_dev, pitch, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  // UV plane (chroma sub-sampled 2×2)
  {
    const int chroma_w = (width + 1) / 2;
    const int chroma_h = (height + 1) / 2;
    const dim3 block(16, 16);
    const dim3 grid((chroma_w + 15) / 16, (chroma_h + 15) / 16);
    rgba_to_nv12_uv_kernel<<<grid, block>>>(rgba_dev, uv_dev, pitch, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  return cudaGetLastError();
}

// ──────────────────────────────────────────────────────────────────────────────
// NV12 letterbox → float32 CHW tensor for TensorRT
// ──────────────────────────────────────────────────────────────────────────────

__global__ void nv12_letterbox_kernel(const uint8_t* y_dev, const uint8_t* uv_dev, int pitch,
                                      float* dst_dev, int src_w, int src_h, int dst_w, int dst_h,
                                      float scale, int pad_x, int pad_y) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;

  const int src_x = static_cast<int>((x - pad_x) / scale);
  const int src_y = static_cast<int>((y - pad_y) / scale);
  const bool in_bounds = src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h;

  float r_f = 0.5f, g_f = 0.5f, b_f = 0.5f;  // letterbox fill (grey)
  if (in_bounds) {
    const int y_raw  = y_dev[src_y * pitch + src_x];
    const int uv_row = src_y / 2;
    const int uv_col = (src_x / 2) * 2;
    const int u_raw  = uv_dev[uv_row * pitch + uv_col];
    const int v_raw  = uv_dev[uv_row * pitch + uv_col + 1];
    uint8_t r, g, b;
    nv12_pixel_to_rgba(y_raw, u_raw, v_raw, &r, &g, &b);
    r_f = r / 255.0f;
    g_f = g / 255.0f;
    b_f = b / 255.0f;
  }

  dst_dev[0 * dst_w * dst_h + y * dst_w + x] = r_f;
  dst_dev[1 * dst_w * dst_h + y * dst_w + x] = g_f;
  dst_dev[2 * dst_w * dst_h + y * dst_w + x] = b_f;
}

extern "C" cudaError_t cuda_nv12_letterbox_device(const uint8_t* y_dev, const uint8_t* uv_dev,
                                                   int pitch, float* dst_dev, int src_w,
                                                   int src_h, int dst_w, int dst_h) {
  const float scale_x = static_cast<float>(dst_w) / src_w;
  const float scale_y = static_cast<float>(dst_h) / src_h;
  const float scale   = scale_x < scale_y ? scale_x : scale_y;
  const int pad_x     = (dst_w - static_cast<int>(src_w * scale)) / 2;
  const int pad_y     = (dst_h - static_cast<int>(src_h * scale)) / 2;

  const dim3 block(16, 16);
  const dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
  nv12_letterbox_kernel<<<grid, block>>>(y_dev, uv_dev, pitch, dst_dev, src_w, src_h, dst_w,
                                        dst_h, scale, pad_x, pad_y);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  return cudaGetLastError();
}

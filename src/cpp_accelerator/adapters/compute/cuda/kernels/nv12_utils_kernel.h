#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Device-memory NV12 ↔ RGBA conversion kernels for in-pipeline GPU-space processing.
// All functions operate on pre-allocated CUDA device buffers — no host copies.
//
// NV12 layout: Y plane (pitch × height bytes) followed by UV plane (pitch × height/2
// bytes, interleaved U0V0 U1V1 …).  The 'pitch' argument is the stride in bytes for
// both planes (they share the same pitch on NVMM surfaces).

// Convert 720p NV12 (NVMM device ptrs) to dense RGBA device buffer.
// Output layout: row-major RGBA, stride = width * 4.
extern "C" cudaError_t cuda_nv12_to_rgba_device(const uint8_t* y_dev, const uint8_t* uv_dev,
                                                 int pitch, uint8_t* rgba_dev, int width,
                                                 int height);

// Convert dense RGBA device buffer back to NV12 device planes.
// Input layout: row-major RGBA, stride = width * 4.
// Output Y / UV pitches = pitch (same as source surface).
extern "C" cudaError_t cuda_rgba_to_nv12_device(const uint8_t* rgba_dev, uint8_t* y_dev,
                                                 uint8_t* uv_dev, int pitch, int width,
                                                 int height);

// Letterbox-resize NV12 device buffer into a pre-allocated float32 device tensor [3×H×W]
// (CHW, normalised to [0,1]) for TensorRT YOLO inference.
// This is the device-pointer companion to cuda_letterbox_resize_to_device; it avoids the
// H→D copy and replaces the NV12→RGB conversion that was previously done on CPU.
extern "C" cudaError_t cuda_nv12_letterbox_device(const uint8_t* y_dev, const uint8_t* uv_dev,
                                                   int pitch, float* dst_dev, int src_w,
                                                   int src_h, int dst_w, int dst_h);

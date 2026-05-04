#pragma once

// NvBufSurface-based NVMM buffer mapping utilities for Jetson (Argus pipeline).
#include <cstdint>
#include <gst/gst.h>

struct NvBufSurface;

namespace jrb::adapters::camera {

// Represents a mapped NVMM NV12 surface.
// NvBufSurfaceMap() returns CPU virtual addresses (host pointers).
// Callers must copy to CUDA device memory before passing to GPU kernels.
struct NvmmFrame {
  uint8_t* y_ptr;         // CPU (host) pointer to Y plane
  uint8_t* uv_ptr;        // CPU (host) pointer to interleaved UV plane
  int width;              // Frame width in pixels
  int height;             // Frame height in pixels
  int pitch;              // Stride in bytes (shared by both planes)
  NvBufSurface* surface;  // Retained for unmapping
};

// Map the NVMM GstBuffer obtained from a `video/x-raw(memory:NVMM)` appsink to
// CUDA-accessible device pointers.  Returns true on success and fills *out.
// The caller MUST call UnmapNvmmBuffer() after CUDA work is complete to release
// the mapping and unmap the GstMapInfo.
bool MapNvmmBuffer(GstBuffer* buf, GstMapInfo* map_info, NvmmFrame* out);

// Release the NVMM mapping acquired by MapNvmmBuffer().
void UnmapNvmmBuffer(GstBuffer* buf, GstMapInfo* map_info);

}  // namespace jrb::adapters::camera

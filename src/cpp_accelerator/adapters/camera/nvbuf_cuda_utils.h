#pragma once

// NvBufSurface-based NVMM buffer mapping utilities for Jetson (Argus pipeline).
// Only compiled when CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED is defined.

#ifdef CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED

#include <cstdint>
#include <gst/gst.h>

struct NvBufSurface;

namespace jrb::adapters::camera {

// Represents a mapped NVMM NV12 surface with CUDA-accessible device pointers.
// Jetson's iGPU uses unified DRAM, so the pointers returned by NvBufSurfaceMap
// are valid as both CPU and CUDA device pointers without any extra copy.
struct NvmmFrame {
  uint8_t* y_ptr;       // Device (and CPU) pointer to Y plane
  uint8_t* uv_ptr;      // Device (and CPU) pointer to interleaved UV plane
  int width;            // Frame width in pixels
  int height;           // Frame height in pixels
  int pitch;            // Stride in bytes (shared by both planes)
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

#endif  // CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED

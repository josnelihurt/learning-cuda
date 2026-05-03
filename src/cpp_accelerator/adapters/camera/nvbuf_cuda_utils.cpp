#include "src/cpp_accelerator/adapters/camera/nvbuf_cuda_utils.h"

#ifdef CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED

#include <nvbufsurface.h>
#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

bool MapNvmmBuffer(GstBuffer* buf, GstMapInfo* map_info, NvmmFrame* out) {
  if (!buf || !map_info || !out) return false;

  // gst_buffer_map on an NVMM buffer: map_info->data points to the NvBufSurface.
  if (!gst_buffer_map(buf, map_info, GST_MAP_READ)) {
    spdlog::error("[NvBufCuda] gst_buffer_map failed");
    return false;
  }

  auto* surface = reinterpret_cast<NvBufSurface*>(map_info->data);
  if (!surface || surface->numFilled == 0) {
    spdlog::error("[NvBufCuda] NvBufSurface is null or empty");
    gst_buffer_unmap(buf, map_info);
    return false;
  }

  // Map the surface planes into CPU/GPU address space.
  // On Jetson iGPU (unified DRAM) this does not trigger a DMA copy.
  if (NvBufSurfaceMap(surface, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
    spdlog::error("[NvBufCuda] NvBufSurfaceMap failed");
    gst_buffer_unmap(buf, map_info);
    return false;
  }

  // Flush CPU cache so the GPU sees the latest sensor data.
  NvBufSurfaceSyncForDevice(surface, 0, -1);

  const NvBufSurfaceParams& params = surface->surfaceList[0];
  out->y_ptr   = static_cast<uint8_t*>(params.mappedAddr.addr[0]);
  out->uv_ptr  = static_cast<uint8_t*>(params.mappedAddr.addr[1]);
  out->width   = static_cast<int>(params.width);
  out->height  = static_cast<int>(params.height);
  out->pitch   = static_cast<int>(params.planeParams.pitch[0]);
  out->surface = surface;
  return true;
}

void UnmapNvmmBuffer(GstBuffer* buf, GstMapInfo* map_info) {
  if (map_info && map_info->data) {
    auto* surface = reinterpret_cast<NvBufSurface*>(map_info->data);
    if (surface && surface->numFilled > 0) {
      // Ensure CPU cache is coherent before unmapping.
      NvBufSurfaceSyncForCpu(surface, 0, -1);
      NvBufSurfaceUnMap(surface, 0, -1);
    }
  }
  if (buf && map_info) {
    gst_buffer_unmap(buf, map_info);
  }
}

}  // namespace jrb::adapters::camera

#endif  // CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED

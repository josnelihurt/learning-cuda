#include "src/cpp_accelerator/adapters/camera/nvbuf_cuda_utils.h"

#include <nvbufsurface.h>
#include <spdlog/spdlog.h>
#include <string_view>

namespace jrb::adapters::camera {
constexpr std::string_view kLogPrefix = "[NvBufCudaUtils]";

bool MapNvmmBuffer(GstBuffer* buf, GstMapInfo* map_info, NvmmFrame* out) {
  if (!buf || !map_info || !out)
    return false;

  // gst_buffer_map on an NVMM buffer: map_info->data points to the NvBufSurface.
  if (!gst_buffer_map(buf, map_info, GST_MAP_READ)) {
    spdlog::error("{} gst_buffer_map failed", kLogPrefix);
    return false;
  }

  auto* surface = reinterpret_cast<NvBufSurface*>(map_info->data);
  if (!surface || surface->numFilled == 0) {
    spdlog::error("{} NvBufSurface is null or empty", kLogPrefix);
    gst_buffer_unmap(buf, map_info);
    return false;
  }

  // Map the surface planes into CPU address space (mmap on the DMA-BUF fd).
  if (NvBufSurfaceMap(surface, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
    spdlog::error("{} NvBufSurfaceMap failed", kLogPrefix);
    gst_buffer_unmap(buf, map_info);
    return false;
  }

  // Argus/ISP wrote the frame data to the NVMM buffer.  Invalidate the CPU
  // cache so that subsequent CPU reads see the device-written pixel data.
  // SyncForCpu = device→CPU direction.  SyncForDevice (CPU→device) would be
  // needed only after CPU writes, which we never do here.
  NvBufSurfaceSyncForCpu(surface, 0, -1);

  const NvBufSurfaceParams& params = surface->surfaceList[0];
  out->y_ptr = static_cast<uint8_t*>(params.mappedAddr.addr[0]);
  out->uv_ptr = static_cast<uint8_t*>(params.mappedAddr.addr[1]);
  out->width = static_cast<int>(params.width);
  out->height = static_cast<int>(params.height);
  out->pitch = static_cast<int>(params.planeParams.pitch[0]);
  out->surface = surface;
  return true;
}

void UnmapNvmmBuffer(GstBuffer* buf, GstMapInfo* map_info) {
  if (map_info && map_info->data) {
    auto* surface = reinterpret_cast<NvBufSurface*>(map_info->data);
    if (surface && surface->numFilled > 0) {
      // We only read from the CPU-mapped buffer (no CPU writes), so no sync
      // is required before unmapping.
      NvBufSurfaceUnMap(surface, 0, -1);
    }
  }
  if (buf && map_info) {
    gst_buffer_unmap(buf, map_info);
  }
}

}  // namespace jrb::adapters::camera

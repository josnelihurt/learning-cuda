/* Minimal NvBufSurface API stub for CI ARM64 compilation.
 *
 * On real Jetson (JetPack 6), the system header at /usr/include/nvbufsurface.h
 * provides the full implementation backed by nvidia-l4t-multimedia-dev.
 * This vendored stub has the same struct layout and function signatures used
 * by nvbuf_cuda_utils.cpp; it enables compilation in CI environments that
 * do not have JetPack packages installed.
 *
 * Source: NVIDIA JetPack 6 / L4T R36 public API
 * https://docs.nvidia.com/jetson/archives/r36.4.3/api_ref/group__NvBufSurface__Group.html
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVBUF_MAX_PLANES 4

typedef enum {
  NVBUF_MAP_READ       = 1,
  NVBUF_MAP_WRITE      = 2,
  NVBUF_MAP_READ_WRITE = 3,
} NvBufSurfaceMemMapFlags;

typedef struct {
  void *addr[NVBUF_MAX_PLANES];
} NvBufSurfaceMappedAddr;

typedef struct {
  uint32_t pitch[NVBUF_MAX_PLANES];
} NvBufSurfacePlaneParams;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t pitch;
  NvBufSurfacePlaneParams planeParams;
  NvBufSurfaceMappedAddr  mappedAddr;
} NvBufSurfaceParams;

typedef struct NvBufSurface {
  int                 gpuId;
  uint32_t            batchSize;
  uint32_t            numFilled;
  NvBufSurfaceParams *surfaceList;
} NvBufSurface;

int NvBufSurfaceMap(NvBufSurface *surf, int index, int plane,
                    NvBufSurfaceMemMapFlags type);
int NvBufSurfaceUnMap(NvBufSurface *surf, int index, int plane);
int NvBufSurfaceSyncForDevice(NvBufSurface *surf, int index, int plane);
int NvBufSurfaceSyncForCpu(NvBufSurface *surf, int index, int plane);

#ifdef __cplusplus
}
#endif

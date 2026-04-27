#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Writes letterbox result into a pre-allocated device buffer (no D→H copy).
extern "C" cudaError_t cuda_letterbox_resize_to_device(const uint8_t* src_host, int src_w,
                                                       int src_h, int src_c, float* dst_device,
                                                       int dst_w, int dst_h);

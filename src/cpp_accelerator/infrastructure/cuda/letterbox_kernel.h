#pragma once
#include <cuda_runtime.h>
#include <cstdint>

extern "C" cudaError_t cuda_letterbox_resize(const uint8_t* src, int src_w, int src_h, int src_c,
                                              float* dst, int dst_w, int dst_h);

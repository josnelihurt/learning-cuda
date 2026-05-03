#pragma once

#include <cuda_runtime.h>

namespace jrb::adapters::compute::cuda {

extern "C" cudaError_t cuda_apply_gaussian_blur_1d_horizontal(const unsigned char* input,
                                                              unsigned char* output, int width,
                                                              int height, int channels,
                                                              const float* kernel, int kernel_size,
                                                              int border_mode, void* pool = nullptr);

extern "C" cudaError_t cuda_apply_gaussian_blur_1d_vertical(const unsigned char* input,
                                                            unsigned char* output, int width,
                                                            int height, int channels,
                                                            const float* kernel, int kernel_size,
                                                            int border_mode, void* pool = nullptr);

extern "C" cudaError_t cuda_apply_gaussian_blur_separable(const unsigned char* input,
                                                          unsigned char* output, int width,
                                                          int height, int channels,
                                                          const float* kernel, int kernel_size,
                                                          int border_mode, void* pool = nullptr);

extern "C" cudaError_t cuda_apply_gaussian_blur_non_separable(const unsigned char* input,
                                                              unsigned char* output, int width,
                                                              int height, int channels,
                                                              const float* kernel, int kernel_size,
                                                              int border_mode, void* pool = nullptr);

// Device-pointer overload: all buffers already reside in device memory.
// rgba_dev is RGBA (4-channel); the blur is applied in-place.  kernel_dev must be a
// pre-uploaded device array of kernel_size floats.  No host copies are performed.
extern "C" cudaError_t cuda_apply_gaussian_blur_separable_device(unsigned char* rgba_dev,
                                                                  int width, int height,
                                                                  const float* kernel_dev,
                                                                  int kernel_size, int border_mode);

}  // namespace jrb::adapters::compute::cuda

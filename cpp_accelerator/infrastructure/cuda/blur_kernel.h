#pragma once

#include <cuda_runtime.h>

namespace jrb::infrastructure::cuda {

extern "C" cudaError_t cuda_apply_gaussian_blur_1d_horizontal(const unsigned char* input,
                                                              unsigned char* output, int width,
                                                              int height, int channels,
                                                              const float* kernel, int kernel_size,
                                                              int border_mode);

extern "C" cudaError_t cuda_apply_gaussian_blur_1d_vertical(const unsigned char* input,
                                                            unsigned char* output, int width,
                                                            int height, int channels,
                                                            const float* kernel, int kernel_size,
                                                            int border_mode);

extern "C" cudaError_t cuda_apply_gaussian_blur_separable(const unsigned char* input,
                                                          unsigned char* output, int width,
                                                          int height, int channels,
                                                          const float* kernel, int kernel_size,
                                                          int border_mode);

extern "C" cudaError_t cuda_apply_gaussian_blur_non_separable(const unsigned char* input,
                                                              unsigned char* output, int width,
                                                              int height, int channels,
                                                              const float* kernel, int kernel_size,
                                                              int border_mode);

}  // namespace jrb::infrastructure::cuda

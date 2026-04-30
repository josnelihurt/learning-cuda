#pragma once

#include <cuda_runtime.h>

namespace jrb::adapters::compute::cuda {

// Pure CUDA function without spdlog dependencies
extern "C" cudaError_t cuda_convert_to_grayscale(const unsigned char* input, unsigned char* output,
                                                 int width, int height, int channels,
                                                 int algorithm);

}  // namespace jrb::adapters::compute::cuda

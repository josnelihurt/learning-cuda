#pragma once

#include <cuda_runtime.h>

namespace jrb::adapters::compute::cuda {

// Pure CUDA function without spdlog dependencies
extern "C" cudaError_t cuda_convert_to_grayscale(const unsigned char* input, unsigned char* output,
                                                 int width, int height, int channels,
                                                 int algorithm);

// Device-pointer overload: converts RGBA device buffer to single-channel grayscale in-place
// (overwrites the R channel; G/B/A untouched).  No host copies.  algorithm follows
// GrayscaleAlgorithmType enum (0 = BT.601).
extern "C" cudaError_t cuda_convert_to_grayscale_device(unsigned char* rgba_dev, int width,
                                                        int height, int algorithm);

}  // namespace jrb::adapters::compute::cuda

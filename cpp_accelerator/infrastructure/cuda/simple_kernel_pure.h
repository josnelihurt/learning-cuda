#pragma once

#include <cuda_runtime.h>

namespace jrb::infrastructure::cuda {

// Pure CUDA function without spdlog dependencies
extern "C" cudaError_t cuda_launch_hello_kernel();

}  // namespace jrb::infrastructure::cuda

#include <cuda_runtime.h>
#include <stdio.h>

namespace jrb::infrastructure::cuda {

__global__ void helloFromGPU() {
  printf("Hello World from GPU! Thread ID: %d, Block ID: %d\n", threadIdx.x, blockIdx.x);
}

// Pure CUDA function without spdlog dependencies
extern "C" cudaError_t cuda_launch_hello_kernel() {
  printf("Hello World from CPU!\n");

  helloFromGPU<<<64, 256>>>();
  cudaError_t error = cudaDeviceSynchronize();

  if (error != cudaSuccess) {
    return error;
  }

  printf("Program completed successfully!\n");
  return cudaSuccess;
}

}  // namespace jrb::infrastructure::cuda

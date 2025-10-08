#include <stdio.h>
#include "lib/cuda/simple_kernel.h"

namespace jrb::lib::cuda {

__global__ void helloFromGPU() {
    printf("Hello World from GPU! Thread ID: %d, Block ID: %d\n", 
           threadIdx.x, blockIdx.x);
}

void launch_hello_kernel() {
    printf("Hello World from CPU!\n");
    
    helloFromGPU<<<64, 256>>>();
    cudaDeviceSynchronize();
    
    printf("Program completed successfully!\n");
}

}  // namespace jrb::lib::cuda

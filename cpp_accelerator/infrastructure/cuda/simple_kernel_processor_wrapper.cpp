#include <spdlog/spdlog.h>
#include "cpp_accelerator/infrastructure/cuda/simple_kernel_processor.h"
#include "cpp_accelerator/infrastructure/cuda/simple_kernel_pure.h"

namespace jrb::infrastructure::cuda {

bool SimpleKernelProcessor::process(domain::interfaces::IImageSource& source,
                                    domain::interfaces::IImageSink& sink,
                                    const std::string& output_path) {
  spdlog::info("Running simple CUDA kernel processor...");

  // Call the pure CUDA function
  cudaError_t error = cuda_launch_hello_kernel();

  if (error != cudaSuccess) {
    spdlog::error("CUDA kernel failed: {}", cudaGetErrorString(error));
    return false;
  }

  spdlog::info("Simple kernel processor completed successfully!");

  // Simple kernel doesn't use source/sink, just runs a hello world kernel
  return true;
}

}  // namespace jrb::infrastructure::cuda

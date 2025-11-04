#include "cpp_accelerator/infrastructure/cuda/simple_kernel_processor.h"
#include "cpp_accelerator/infrastructure/cuda/simple_kernel_pure.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::infrastructure::cuda {

// Simple test implementation that runs a hello world CUDA kernel
// This implementation doesn't process images, so source/sink/output_path are not used
bool SimpleKernelProcessor::process([[maybe_unused]] domain::interfaces::IImageSource& source,
                                    [[maybe_unused]] domain::interfaces::IImageSink& sink,
                                    [[maybe_unused]] const std::string& output_path) {
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

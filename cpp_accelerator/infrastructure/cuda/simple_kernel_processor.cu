#include "cpp_accelerator/infrastructure/cuda/simple_kernel_processor.h"
#include "cpp_accelerator/infrastructure/cuda/simple_kernel.h"
#include <spdlog/spdlog.h>

namespace jrb::infrastructure::cuda {

bool SimpleKernelProcessor::process(domain::interfaces::IImageSource& source, 
                                   domain::interfaces::IImageSink& sink,
                                   const std::string& output_path) {
  spdlog::info("Running simple CUDA kernel processor...");
  
  // Launch the simple hello kernel
  launch_hello_kernel();
  
  spdlog::info("Simple kernel processor completed successfully!");
  
  // Simple kernel doesn't use source/sink, just runs a hello world kernel
  return true;
}

}  // namespace jrb::infrastructure::cuda


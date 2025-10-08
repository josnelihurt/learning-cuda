#include "app/simple_kernel_command.h"
#include "lib/cuda/simple_kernel.h"
#include <spdlog/spdlog.h>

namespace jrb::app {

core::Result<void> SimpleKernelCommand::execute() {
    spdlog::info("Running simple CUDA kernel...");
    
    jrb::lib::cuda::launch_hello_kernel();
    
    spdlog::info("Simple kernel completed successfully!");
    
    return core::Result<void>::ok("Simple kernel execution completed", 0);
}

}  // namespace jrb::app


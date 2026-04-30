#include "src/cpp_accelerator/application/server_info/accelerator_label.h"

namespace jrb::application::server_info {

std::string AcceleratorTypeLabel(cuda_learning::AcceleratorType type) {
  switch (type) {
    case cuda_learning::ACCELERATOR_TYPE_CUDA:   return "CUDA";
    case cuda_learning::ACCELERATOR_TYPE_CPU:    return "CPU";
    case cuda_learning::ACCELERATOR_TYPE_OPENCL: return "OpenCL";
    case cuda_learning::ACCELERATOR_TYPE_VULKAN: return "Vulkan";
    default:                                     return "Unknown";
  }
}

}  // namespace jrb::application::server_info

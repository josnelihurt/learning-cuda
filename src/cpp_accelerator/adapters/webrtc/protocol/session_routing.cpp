#include "src/cpp_accelerator/adapters/webrtc/protocol/session_routing.h"

namespace jrb::adapters::webrtc::protocol {

bool IsGoVideoSession(const std::string& value) {
  return value.rfind(kGoVideoSessionPrefix, 0) == 0;
}

bool ShouldRegisterSessionChannel(const std::string& session_id, const std::string& label) {
  return !IsGoVideoSession(session_id) && !IsGoVideoSession(label);
}

std::string AcceleratorTypeLabel(cuda_learning::AcceleratorType type) {
  switch (type) {
    case cuda_learning::ACCELERATOR_TYPE_CUDA:   return "CUDA";
    case cuda_learning::ACCELERATOR_TYPE_CPU:    return "CPU";
    case cuda_learning::ACCELERATOR_TYPE_OPENCL: return "OpenCL";
    case cuda_learning::ACCELERATOR_TYPE_VULKAN: return "Vulkan";
    default:                                     return "Unknown";
  }
}

}  // namespace jrb::adapters::webrtc::protocol

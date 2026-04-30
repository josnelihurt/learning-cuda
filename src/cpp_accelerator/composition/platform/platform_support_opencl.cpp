#include "src/cpp_accelerator/composition/platform/platform_support.h"

#include "src/cpp_accelerator/composition/platform/cpu/cpu_platform.h"
#include "src/cpp_accelerator/composition/platform/opencl/opencl_platform.h"

namespace jrb::application::engine {

void RegisterPlatformAccelerators(FilterFactoryRegistry& registry) {
  platform::cpu::RegisterFactories(registry);
  platform::opencl::RegisterFactories(registry);
}

void InitializePlatformSubsystems(const cuda_learning::InitRequest&, cuda_learning::InitResponse*) {
}

bool ApplyInference(const std::string&, float, bool, const cuda_learning::ProcessImageRequest&,
                    cuda_learning::ProcessImageResponse*,
                    const jrb::domain::interfaces::ImageBuffer&,
                    jrb::domain::interfaces::ImageBufferMut&) {
  return false;
}

}  // namespace jrb::application::engine

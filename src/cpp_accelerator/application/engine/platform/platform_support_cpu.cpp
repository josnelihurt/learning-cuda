#include "src/cpp_accelerator/application/engine/platform/platform_support.h"

#include "src/cpp_accelerator/adapters/compute/cpu/cpu_filter_factory.h"

namespace jrb::application::engine {

void RegisterPlatformAccelerators(FilterFactoryRegistry& registry) {
  registry.Register(std::make_unique<jrb::infrastructure::cpu::CpuFilterFactory>());
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

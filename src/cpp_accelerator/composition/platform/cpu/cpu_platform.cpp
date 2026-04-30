#include "src/cpp_accelerator/composition/platform/cpu/cpu_platform.h"

#include <memory>
#include "src/cpp_accelerator/adapters/compute/cpu/cpu_filter_factory.h"

namespace jrb::application::engine::platform::cpu {

void RegisterFactories(FilterFactoryRegistry& registry) {
  registry.Register(std::make_unique<jrb::adapters::compute::cpu::CpuFilterFactory>());
}

}  // namespace jrb::application::engine::platform::cpu

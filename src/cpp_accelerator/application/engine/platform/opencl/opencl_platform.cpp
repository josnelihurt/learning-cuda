#include "src/cpp_accelerator/application/engine/platform/opencl/opencl_platform.h"

#include <memory>
#include "src/cpp_accelerator/adapters/compute/opencl/opencl_filter_factory.h"

namespace jrb::application::engine::platform::opencl {

void RegisterFactories(FilterFactoryRegistry& registry) {
  registry.Register(std::make_unique<jrb::infrastructure::opencl::OpenCLFilterFactory>());
}

}  // namespace jrb::application::engine::platform::opencl

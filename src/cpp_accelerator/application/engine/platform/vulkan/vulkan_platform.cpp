#include "src/cpp_accelerator/application/engine/platform/vulkan/vulkan_platform.h"

#include <memory>
#include "src/cpp_accelerator/adapters/compute/vulkan/vulkan_filter_factory.h"

namespace jrb::application::engine::platform::vulkan {

void RegisterFactories(FilterFactoryRegistry& registry) {
  registry.Register(std::make_unique<jrb::infrastructure::vulkan::VulkanFilterFactory>());
}

}  // namespace jrb::application::engine::platform::vulkan

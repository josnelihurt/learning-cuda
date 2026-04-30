#include "src/cpp_accelerator/adapters/compute/vulkan/vulkan_filter_factory.h"

#include "src/cpp_accelerator/adapters/compute/vulkan/filters/blur_filter.h"
#include "src/cpp_accelerator/adapters/compute/vulkan/filters/grayscale_filter.h"
#include "src/cpp_accelerator/application/engine/filter_creation_dispatch.hpp"
#include "src/cpp_accelerator/application/engine/filter_descriptor.h"

namespace jrb::adapters::compute::vulkan {

using jrb::application::engine::FilterCreationParams;
using jrb::application::engine::FilterDescriptor;
using jrb::domain::interfaces::FilterType;

cuda_learning::AcceleratorType VulkanFilterFactory::GetAcceleratorType() const {
  return cuda_learning::ACCELERATOR_TYPE_VULKAN;
}

std::vector<FilterDescriptor> VulkanFilterFactory::GetFilterDescriptors() const {
  // No user-configurable parameters — Vulkan applies BT.601 grayscale and fixed 5×5 σ≈1 blur.
  return {
      FilterDescriptor{.id = "grayscale", .name = "Grayscale", .parameters = {}},
      FilterDescriptor{.id = "blur", .name = "Gaussian Blur", .parameters = {}},
  };
}

std::unique_ptr<jrb::domain::interfaces::IFilter> VulkanFilterFactory::CreateFilter(
    FilterType type, const FilterCreationParams& /*params*/) const {
  return jrb::application::engine::DispatchCreateFilter(
      type, []() { return std::make_unique<GrayscaleFilter>(); },
      []() { return std::make_unique<GaussianBlurFilter>(); });
}

}  // namespace jrb::adapters::compute::vulkan

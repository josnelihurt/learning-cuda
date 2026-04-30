#pragma once

#include "src/cpp_accelerator/application/engine/i_filter_factory.h"

namespace jrb::infrastructure::vulkan {

// Filter factory for the Vulkan compute accelerator.
// Supports grayscale (BT.601) and Gaussian blur (5×5, σ≈1) via SPIR-V compute shaders.
// Filters have no user-configurable parameters.
class VulkanFilterFactory : public jrb::application::engine::IFilterFactory {
 public:
  VulkanFilterFactory() = default;
  ~VulkanFilterFactory() override = default;

  cuda_learning::AcceleratorType GetAcceleratorType() const override;

  std::vector<jrb::application::engine::FilterDescriptor> GetFilterDescriptors() const override;

  std::unique_ptr<jrb::domain::interfaces::IFilter> CreateFilter(
      jrb::domain::interfaces::FilterType type,
      const jrb::application::engine::FilterCreationParams& params) const override;
};

}  // namespace jrb::infrastructure::vulkan

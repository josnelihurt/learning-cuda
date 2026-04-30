#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <vulkan/vulkan.hpp>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"

namespace jrb::infrastructure::vulkan {

// Applies a separable 5-tap Gaussian blur (sigma≈1, weights=[1,4,6,4,1]/16) using
// two Vulkan compute dispatches: horizontal then vertical.
class VulkanBlurFilter : public jrb::domain::interfaces::IFilter {
 public:
  VulkanBlurFilter();
  ~VulkanBlurFilter() override;

  bool Apply(jrb::domain::interfaces::FilterContext& context) override;
  jrb::domain::interfaces::FilterType GetType() const override;
  bool IsInPlace() const override;

 private:
  bool EnsurePipeline();
  void DestroyPipeline();

  bool AllocateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::Buffer& buf_out,
                      vk::DeviceMemory& mem_out);

  bool pipeline_ready_;
  vk::DescriptorSetLayout descriptor_set_layout_;
  vk::PipelineLayout pipeline_layout_;
  vk::ShaderModule shader_module_;
  vk::Pipeline pipeline_;
};

}  // namespace jrb::infrastructure::vulkan

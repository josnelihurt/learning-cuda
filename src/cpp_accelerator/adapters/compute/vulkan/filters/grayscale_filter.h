#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <vulkan/vulkan.hpp>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"

namespace jrb::adapters::compute::vulkan {

// Converts an RGB image to single-channel grayscale using BT.601 luma
// coefficients (L = 0.299R + 0.587G + 0.114B) on the Vulkan compute device.
class GrayscaleFilter : public jrb::domain::interfaces::IFilter {
 public:
  GrayscaleFilter();
  ~GrayscaleFilter() override;

  bool Apply(jrb::domain::interfaces::FilterContext& context) override;
  jrb::domain::interfaces::FilterType GetType() const override;
  bool IsInPlace() const override;

 private:
  bool EnsurePipeline();
  void DestroyPipeline();

  // Allocate a host-visible, host-coherent device buffer.
  bool AllocateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::Buffer& buf_out,
                      vk::DeviceMemory& mem_out);

  bool pipeline_ready_;
  vk::DescriptorSetLayout descriptor_set_layout_;
  vk::PipelineLayout pipeline_layout_;
  vk::ShaderModule shader_module_;
  vk::Pipeline pipeline_;
};

}  // namespace jrb::adapters::compute::vulkan

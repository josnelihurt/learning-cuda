#include "src/cpp_accelerator/adapters/compute/vulkan/vulkan_grayscale_filter.h"

#include <cstddef>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/adapters/compute/vulkan/vk_grayscale_kernel_blob.h"
#include "src/cpp_accelerator/adapters/compute/vulkan/vulkan_compute_utils.h"
#include "src/cpp_accelerator/adapters/compute/vulkan/vulkan_context.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::infrastructure::vulkan {

VulkanGrayscaleFilter::VulkanGrayscaleFilter()
    : pipeline_ready_(false) {}

VulkanGrayscaleFilter::~VulkanGrayscaleFilter() { DestroyPipeline(); }

bool VulkanGrayscaleFilter::EnsurePipeline() {
  if (pipeline_ready_) return true;

  auto& ctx = VulkanContext::GetInstance();
  if (!ctx.available()) {
    spdlog::error("[VulkanGrayscale] context unavailable: {}", ctx.error_message());
    return false;
  }

  vk::Device device = ctx.device();

  // Create shader module from embedded SPIR-V.
  const auto* spirv = reinterpret_cast<const uint32_t*>(vk_grayscale_kernel_blob::spirv());
  size_t spirv_size = vk_grayscale_kernel_blob::spirv_size_bytes();
  if (spirv == nullptr || spirv_size == 0 || spirv_size % sizeof(uint32_t) != 0) {
    spdlog::error("[VulkanGrayscale] invalid embedded SPIR-V");
    return false;
  }
  try {
    shader_module_ = device.createShaderModule(vk::ShaderModuleCreateInfo({}, spirv_size, spirv));
  } catch (const vk::SystemError& e) {
    spdlog::error("[VulkanGrayscale] createShaderModule failed: {}", e.what());
    return false;
  }

  // Descriptor set layout: two storage buffers (input + output).
  std::vector<vk::DescriptorSetLayoutBinding> bindings = {
      {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
      {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
  };
  try {
    descriptor_set_layout_ =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, bindings));
  } catch (const vk::SystemError& e) {
    spdlog::error("[VulkanGrayscale] createDescriptorSetLayout failed: {}", e.what());
    DestroyPipeline();
    return false;
  }

  // Push constants: width, height (two uint32s).
  vk::PushConstantRange push_range(vk::ShaderStageFlagBits::eCompute, 0, 2 * sizeof(uint32_t));
  try {
    pipeline_layout_ = device.createPipelineLayout(
        vk::PipelineLayoutCreateInfo({}, 1, &descriptor_set_layout_, 1, &push_range));
  } catch (const vk::SystemError& e) {
    spdlog::error("[VulkanGrayscale] createPipelineLayout failed: {}", e.what());
    DestroyPipeline();
    return false;
  }

  vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, shader_module_,
                                          "main");
  try {
    auto result = device.createComputePipeline(
        nullptr, vk::ComputePipelineCreateInfo({}, stage, pipeline_layout_));
    pipeline_ = result.value;
  } catch (const vk::SystemError& e) {
    spdlog::error("[VulkanGrayscale] createComputePipeline failed: {}", e.what());
    DestroyPipeline();
    return false;
  }

  pipeline_ready_ = true;
  return true;
}

void VulkanGrayscaleFilter::DestroyPipeline() {
  auto& ctx = VulkanContext::GetInstance();
  if (!ctx.available()) return;
  vk::Device device = ctx.device();
  if (pipeline_) {
    device.destroyPipeline(pipeline_);
    pipeline_ = nullptr;
  }
  if (pipeline_layout_) {
    device.destroyPipelineLayout(pipeline_layout_);
    pipeline_layout_ = nullptr;
  }
  if (descriptor_set_layout_) {
    device.destroyDescriptorSetLayout(descriptor_set_layout_);
    descriptor_set_layout_ = nullptr;
  }
  if (shader_module_) {
    device.destroyShaderModule(shader_module_);
    shader_module_ = nullptr;
  }
  pipeline_ready_ = false;
}

bool VulkanGrayscaleFilter::Apply(jrb::domain::interfaces::FilterContext& context) {
  if (!EnsurePipeline()) return false;

  auto& ctx = VulkanContext::GetInstance();
  vk::Device device = ctx.device();

  const int width = context.input.width;
  const int height = context.input.height;
  const int channels = context.input.channels;

  if (channels != 3) {
    spdlog::error("[VulkanGrayscale] expected 3-channel input, got {}", channels);
    return false;
  }

  const size_t n_in = static_cast<size_t>(width) * height * channels;
  const size_t n_out = static_cast<size_t>(width) * height;
  const vk::DeviceSize in_bytes = n_in * sizeof(float);
  const vk::DeviceSize out_bytes = n_out * sizeof(float);

  // Allocate host-visible buffers.
  vk::Buffer in_buf, out_buf;
  vk::DeviceMemory in_mem, out_mem;
  if (!AllocateHostBuffer(device, ctx.physical_device(), in_bytes,
                          vk::BufferUsageFlagBits::eStorageBuffer, in_buf, in_mem)) {
    return false;
  }
  if (!AllocateHostBuffer(device, ctx.physical_device(), out_bytes,
                          vk::BufferUsageFlagBits::eStorageBuffer, out_buf, out_mem)) {
    device.destroyBuffer(in_buf);
    device.freeMemory(in_mem);
    return false;
  }

  // Upload: widen bytes to floats.
  {
    auto* mapped = reinterpret_cast<float*>(device.mapMemory(in_mem, 0, in_bytes));
    BytesToFloats(context.input.data, mapped, n_in);
    device.unmapMemory(in_mem);
  }

  // Allocate and record command buffer.
  vk::CommandBuffer cmd;
  try {
    auto cmds = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(ctx.command_pool(), vk::CommandBufferLevel::ePrimary, 1));
    cmd = cmds[0];
  } catch (const vk::SystemError& e) {
    spdlog::error("[VulkanGrayscale] allocateCommandBuffers failed: {}", e.what());
    device.destroyBuffer(in_buf);
    device.freeMemory(in_mem);
    device.destroyBuffer(out_buf);
    device.freeMemory(out_mem);
    return false;
  }

  // Descriptor pool + set.
  vk::DescriptorPool desc_pool;
  vk::DescriptorSet desc_set;
  std::array<vk::DescriptorPoolSize, 1> pool_sizes = {
      vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2)};
  try {
    desc_pool = device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo({}, 1, pool_sizes.size(), pool_sizes.data()));
    auto sets = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo(desc_pool, 1, &descriptor_set_layout_));
    desc_set = sets[0];
  } catch (const vk::SystemError& e) {
    spdlog::error("[VulkanGrayscale] descriptor allocation failed: {}", e.what());
    device.freeCommandBuffers(ctx.command_pool(), cmd);
    device.destroyBuffer(in_buf);
    device.freeMemory(in_mem);
    device.destroyBuffer(out_buf);
    device.freeMemory(out_mem);
    return false;
  }

  vk::DescriptorBufferInfo in_info(in_buf, 0, in_bytes);
  vk::DescriptorBufferInfo out_info(out_buf, 0, out_bytes);
  std::array<vk::WriteDescriptorSet, 2> writes = {
      vk::WriteDescriptorSet{desc_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
                             &in_info},
      vk::WriteDescriptorSet{desc_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
                             &out_info},
  };
  device.updateDescriptorSets(writes, {});

  struct PushData {
    uint32_t width;
    uint32_t height;
  } push{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

  cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout_, 0, desc_set, {});
  cmd.pushConstants(pipeline_layout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
  uint32_t gx = (static_cast<uint32_t>(width) + 15u) / 16u;
  uint32_t gy = (static_cast<uint32_t>(height) + 15u) / 16u;
  cmd.dispatch(gx, gy, 1);

  bool ok = SubmitAndWait(device, ctx.queue(), cmd, "VulkanGrayscale");

  // Readback: narrow floats to bytes.
  if (ok) {
    const auto* mapped = reinterpret_cast<const float*>(device.mapMemory(out_mem, 0, out_bytes));
    FloatsToBytes(mapped, context.output.data, n_out);
    device.unmapMemory(out_mem);
  }

  device.freeCommandBuffers(ctx.command_pool(), cmd);
  device.destroyDescriptorPool(desc_pool);
  device.destroyBuffer(in_buf);
  device.freeMemory(in_mem);
  device.destroyBuffer(out_buf);
  device.freeMemory(out_mem);

  return ok;
}

jrb::domain::interfaces::FilterType VulkanGrayscaleFilter::GetType() const {
  return jrb::domain::interfaces::FilterType::GRAYSCALE;
}

bool VulkanGrayscaleFilter::IsInPlace() const { return false; }

}  // namespace jrb::infrastructure::vulkan

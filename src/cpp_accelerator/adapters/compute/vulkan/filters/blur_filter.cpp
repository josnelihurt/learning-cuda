#include "src/cpp_accelerator/adapters/compute/vulkan/filters/blur_filter.h"

#include <cstddef>
#include <string_view>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/adapters/compute/vulkan/context/compute_utils.h"
#include "src/cpp_accelerator/adapters/compute/vulkan/context/context.h"
#include "src/cpp_accelerator/adapters/compute/vulkan/kernels/vk_blur_blob.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::adapters::compute::vulkan {
namespace {
constexpr std::string_view kLogPrefix = "[VulkanBlur]";
}

GaussianBlurFilter::GaussianBlurFilter() : pipeline_ready_(false) {}

GaussianBlurFilter::~GaussianBlurFilter() {
  DestroyPipeline();
}

bool GaussianBlurFilter::EnsurePipeline() {
  if (pipeline_ready_)
    return true;

  auto& ctx = Context::GetInstance();
  if (!ctx.available()) {
    spdlog::error("{} context unavailable: {}", kLogPrefix, ctx.error_message());
    return false;
  }

  vk::Device device = ctx.device();

  const auto* spirv = reinterpret_cast<const uint32_t*>(vk_blur_blob::spirv());
  size_t spirv_size = vk_blur_blob::spirv_size_bytes();
  if (spirv == nullptr || spirv_size == 0 || spirv_size % sizeof(uint32_t) != 0) {
    spdlog::error("{} invalid embedded SPIR-V", kLogPrefix, kLogPrefix);
    return false;
  }
  try {
    shader_module_ = device.createShaderModule(vk::ShaderModuleCreateInfo({}, spirv_size, spirv));
  } catch (const vk::SystemError& e) {
    spdlog::error("{} createShaderModule failed: {}", kLogPrefix, e.what());
    return false;
  }

  std::vector<vk::DescriptorSetLayoutBinding> bindings = {
      {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
      {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
  };
  try {
    descriptor_set_layout_ =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, bindings));
  } catch (const vk::SystemError& e) {
    spdlog::error("{} createDescriptorSetLayout failed: {}", kLogPrefix, e.what());
    DestroyPipeline();
    return false;
  }

  // Push constants: width, height, channels, pass (4 × uint32).
  vk::PushConstantRange push_range(vk::ShaderStageFlagBits::eCompute, 0, 4 * sizeof(uint32_t));
  try {
    pipeline_layout_ = device.createPipelineLayout(
        vk::PipelineLayoutCreateInfo({}, 1, &descriptor_set_layout_, 1, &push_range));
  } catch (const vk::SystemError& e) {
    spdlog::error("{} createPipelineLayout failed: {}", kLogPrefix, e.what());
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
    spdlog::error("{} createComputePipeline failed: {}", kLogPrefix, e.what());
    DestroyPipeline();
    return false;
  }

  pipeline_ready_ = true;
  return true;
}

void GaussianBlurFilter::DestroyPipeline() {
  auto& ctx = Context::GetInstance();
  if (!ctx.available())
    return;
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

// Runs one pass of the blur shader (h or v).
// src_buf/src_mem → dst_buf/dst_mem.  Both already bound with the given buffer infos.
static bool RunBlurPass(vk::Device device, vk::Queue queue, vk::CommandPool cmd_pool,
                        vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout,
                        vk::DescriptorSetLayout desc_set_layout, vk::DescriptorPool desc_pool,
                        vk::Buffer src_buf, vk::DeviceSize src_bytes, vk::Buffer dst_buf,
                        vk::DeviceSize dst_bytes, uint32_t width, uint32_t height,
                        uint32_t channels, uint32_t pass) {
  // Allocate a fresh descriptor set for this pass (different src/dst each pass).
  vk::DescriptorSet desc_set;
  try {
    auto sets = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo(desc_pool, 1, &desc_set_layout));
    desc_set = sets[0];
  } catch (const vk::SystemError& e) {
    spdlog::error("{} allocateDescriptorSets pass={} failed: {}", kLogPrefix, pass, e.what());
    return false;
  }

  vk::DescriptorBufferInfo src_info(src_buf, 0, src_bytes);
  vk::DescriptorBufferInfo dst_info(dst_buf, 0, dst_bytes);
  std::array<vk::WriteDescriptorSet, 2> writes = {
      vk::WriteDescriptorSet{desc_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
                             &src_info},
      vk::WriteDescriptorSet{desc_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,
                             &dst_info},
  };
  device.updateDescriptorSets(writes, {});

  struct PushData {
    uint32_t width, height, channels, pass;
  } push{width, height, channels, pass};

  vk::CommandBuffer cmd;
  try {
    auto cmds = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(cmd_pool, vk::CommandBufferLevel::ePrimary, 1));
    cmd = cmds[0];
  } catch (const vk::SystemError& e) {
    spdlog::error("{} allocateCommandBuffers pass={} failed: {}", kLogPrefix, pass, e.what());
    return false;
  }

  cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, desc_set, {});
  cmd.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
  uint32_t gx = (width + 15u) / 16u;
  uint32_t gy = (height + 15u) / 16u;
  cmd.dispatch(gx, gy, 1);

  bool ok = SubmitAndWait(device, queue, cmd, "VulkanBlur");
  device.freeCommandBuffers(cmd_pool, cmd);
  return ok;
}

bool GaussianBlurFilter::Apply(jrb::domain::interfaces::FilterContext& context) {
  if (!EnsurePipeline())
    return false;

  auto& ctx = Context::GetInstance();
  vk::Device device = ctx.device();

  const uint32_t width = static_cast<uint32_t>(context.input.width);
  const uint32_t height = static_cast<uint32_t>(context.input.height);
  const uint32_t channels = static_cast<uint32_t>(context.input.channels);
  const size_t n_samples = static_cast<size_t>(width) * height * channels;
  const vk::DeviceSize buf_bytes = n_samples * sizeof(float);

  // Three buffers: src (input), tmp (h-pass output / v-pass input), dst (output).
  vk::Buffer src_buf, tmp_buf, dst_buf;
  vk::DeviceMemory src_mem, tmp_mem, dst_mem;

  auto cleanup = [&]() {
    if (src_buf) {
      device.destroyBuffer(src_buf);
      device.freeMemory(src_mem);
    }
    if (tmp_buf) {
      device.destroyBuffer(tmp_buf);
      device.freeMemory(tmp_mem);
    }
    if (dst_buf) {
      device.destroyBuffer(dst_buf);
      device.freeMemory(dst_mem);
    }
  };

  if (!AllocateHostBuffer(device, ctx.physical_device(), buf_bytes,
                          vk::BufferUsageFlagBits::eStorageBuffer, src_buf, src_mem) ||
      !AllocateHostBuffer(device, ctx.physical_device(), buf_bytes,
                          vk::BufferUsageFlagBits::eStorageBuffer, tmp_buf, tmp_mem) ||
      !AllocateHostBuffer(device, ctx.physical_device(), buf_bytes,
                          vk::BufferUsageFlagBits::eStorageBuffer, dst_buf, dst_mem)) {
    cleanup();
    return false;
  }

  // Upload input.
  {
    auto* mapped = reinterpret_cast<float*>(device.mapMemory(src_mem, 0, buf_bytes));
    BytesToFloats(context.input.data, mapped, n_samples);
    device.unmapMemory(src_mem);
  }

  // Descriptor pool: 2 passes × 2 bindings = 4 descriptors.
  vk::DescriptorPool desc_pool;
  std::array<vk::DescriptorPoolSize, 1> pool_sizes = {
      vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4)};
  try {
    desc_pool = device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 2,
                                     pool_sizes.size(), pool_sizes.data()));
  } catch (const vk::SystemError& e) {
    spdlog::error("{} createDescriptorPool failed: {}", kLogPrefix, e.what());
    cleanup();
    return false;
  }

  bool ok = RunBlurPass(device, ctx.queue(), ctx.command_pool(), pipeline_, pipeline_layout_,
                        descriptor_set_layout_, desc_pool, src_buf, buf_bytes, tmp_buf, buf_bytes,
                        width, height, channels, 0u);
  if (ok) {
    ok = RunBlurPass(device, ctx.queue(), ctx.command_pool(), pipeline_, pipeline_layout_,
                     descriptor_set_layout_, desc_pool, tmp_buf, buf_bytes, dst_buf, buf_bytes,
                     width, height, channels, 1u);
  }

  if (ok) {
    const auto* mapped = reinterpret_cast<const float*>(device.mapMemory(dst_mem, 0, buf_bytes));
    FloatsToBytes(mapped, context.output.data, n_samples);
    device.unmapMemory(dst_mem);
  }

  device.destroyDescriptorPool(desc_pool);
  cleanup();
  return ok;
}

jrb::domain::interfaces::FilterType GaussianBlurFilter::GetType() const {
  return jrb::domain::interfaces::FilterType::BLUR;
}

bool GaussianBlurFilter::IsInPlace() const {
  return false;
}

}  // namespace jrb::adapters::compute::vulkan

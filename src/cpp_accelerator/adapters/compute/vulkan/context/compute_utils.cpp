#include "src/cpp_accelerator/adapters/compute/vulkan/context/compute_utils.h"

#include <string_view>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::adapters::compute::vulkan {

namespace {
constexpr std::string_view kLogPrefix = "[VulkanCompute]";
}

uint32_t FindHostMemoryType(vk::PhysicalDevice pd, const vk::MemoryRequirements& reqs) {
  auto props = pd.getMemoryProperties();
  constexpr vk::MemoryPropertyFlags kNeeded =
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if (((reqs.memoryTypeBits & (1u << i)) != 0u) &&
        ((props.memoryTypes[i].propertyFlags & kNeeded) == kNeeded)) {
      return i;
    }
  }
  return 0xFFFFFFFFu;
}

bool AllocateHostBuffer(vk::Device device, vk::PhysicalDevice pd, vk::DeviceSize size,
                        vk::BufferUsageFlags usage, vk::Buffer& buf_out,
                        vk::DeviceMemory& mem_out) {
  try {
    buf_out =
        device.createBuffer(vk::BufferCreateInfo({}, size, usage, vk::SharingMode::eExclusive));
  } catch (const vk::SystemError& e) {
    spdlog::error("{} createBuffer failed: {}", kLogPrefix, e.what());
    return false;
  }

  vk::MemoryRequirements reqs = device.getBufferMemoryRequirements(buf_out);
  uint32_t type_index = FindHostMemoryType(pd, reqs);
  if (type_index == 0xFFFFFFFFu) {
    spdlog::error("{} no suitable host-visible memory type", kLogPrefix);
    device.destroyBuffer(buf_out);
    buf_out = nullptr;
    return false;
  }

  try {
    mem_out = device.allocateMemory(vk::MemoryAllocateInfo(reqs.size, type_index));
  } catch (const vk::SystemError& e) {
    spdlog::error("{} allocateMemory failed: {}", kLogPrefix, e.what());
    device.destroyBuffer(buf_out);
    buf_out = nullptr;
    return false;
  }

  device.bindBufferMemory(buf_out, mem_out, 0);
  return true;
}

void BytesToFloats(const unsigned char* src, float* dst, size_t n) {
  for (size_t i = 0; i < n; ++i)
    dst[i] = static_cast<float>(src[i]);
}

void FloatsToBytes(const float* src, unsigned char* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float v = src[i] + 0.5f;
    if (v < 0.0f)
      v = 0.0f;
    if (v > 255.0f)
      v = 255.0f;
    dst[i] = static_cast<unsigned char>(v);
  }
}

bool SubmitAndWait(vk::Device device, vk::Queue queue, vk::CommandBuffer cmd,
                   const char* tag) {
  cmd.end();
  vk::SubmitInfo submit_info(0, nullptr, nullptr, 1, &cmd);
  vk::Fence fence;
  try {
    fence = device.createFence(vk::FenceCreateInfo());
  } catch (const vk::SystemError& e) {
    spdlog::error("[{}] createFence failed: {}", tag, e.what());
    return false;
  }
  queue.submit(submit_info, fence);
  (void)device.waitForFences(fence, vk::True, UINT64_MAX);
  device.destroyFence(fence);
  return true;
}

}  // namespace jrb::adapters::compute::vulkan

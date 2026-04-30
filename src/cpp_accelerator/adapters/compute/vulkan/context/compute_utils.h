#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/adapters/compute/vulkan/context/context.h"

namespace jrb::adapters::compute::vulkan {

// Finds a host-visible, host-coherent memory type index for the given requirements.
inline uint32_t FindHostMemoryType(vk::PhysicalDevice pd, const vk::MemoryRequirements& reqs) {
  auto props = pd.getMemoryProperties();
  constexpr vk::MemoryPropertyFlags kNeeded = vk::MemoryPropertyFlagBits::eHostVisible |
                                              vk::MemoryPropertyFlagBits::eHostCoherent;
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if (((reqs.memoryTypeBits & (1u << i)) != 0u) &&
        ((props.memoryTypes[i].propertyFlags & kNeeded) == kNeeded)) {
      return i;
    }
  }
  return 0xFFFFFFFFu;
}

// Allocates a host-visible, host-coherent Vulkan buffer backed by device memory.
inline bool AllocateHostBuffer(vk::Device device, vk::PhysicalDevice pd, vk::DeviceSize size,
                               vk::BufferUsageFlags usage, vk::Buffer& buf_out,
                               vk::DeviceMemory& mem_out) {
  try {
    buf_out = device.createBuffer(
        vk::BufferCreateInfo({}, size, usage, vk::SharingMode::eExclusive));
  } catch (const vk::SystemError& e) {
    spdlog::error("[VulkanCompute] createBuffer failed: {}", e.what());
    return false;
  }

  vk::MemoryRequirements reqs = device.getBufferMemoryRequirements(buf_out);
  uint32_t type_index = FindHostMemoryType(pd, reqs);
  if (type_index == 0xFFFFFFFFu) {
    spdlog::error("[VulkanCompute] no suitable host-visible memory type");
    device.destroyBuffer(buf_out);
    buf_out = nullptr;
    return false;
  }

  try {
    mem_out = device.allocateMemory(vk::MemoryAllocateInfo(reqs.size, type_index));
  } catch (const vk::SystemError& e) {
    spdlog::error("[VulkanCompute] allocateMemory failed: {}", e.what());
    device.destroyBuffer(buf_out);
    buf_out = nullptr;
    return false;
  }

  device.bindBufferMemory(buf_out, mem_out, 0);
  return true;
}

// Widens a byte array to a float array (values remain in [0, 255]).
inline void BytesToFloats(const unsigned char* src, float* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
}

// Narrows a float array to a byte array (clamps to [0, 255], rounds to nearest).
inline void FloatsToBytes(const float* src, unsigned char* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float v = src[i] + 0.5f;
    if (v < 0.0f) v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    dst[i] = static_cast<unsigned char>(v);
  }
}

// Submits a one-shot command buffer and waits for completion.
inline bool SubmitAndWait(vk::Device device, vk::Queue queue,
                          vk::CommandBuffer cmd, const char* tag) {
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

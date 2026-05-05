#pragma once

#include <cstddef>
#include <cstdint>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <vulkan/vulkan.hpp>
#pragma GCC diagnostic pop

namespace jrb::adapters::compute::vulkan {

uint32_t FindHostMemoryType(vk::PhysicalDevice pd, const vk::MemoryRequirements& reqs);

bool AllocateHostBuffer(vk::Device device, vk::PhysicalDevice pd, vk::DeviceSize size,
                        vk::BufferUsageFlags usage, vk::Buffer& buf_out,
                        vk::DeviceMemory& mem_out);

void BytesToFloats(const unsigned char* src, float* dst, size_t n);

void FloatsToBytes(const float* src, unsigned char* dst, size_t n);

bool SubmitAndWait(vk::Device device, vk::Queue queue, vk::CommandBuffer cmd,
                   const char* tag);

}  // namespace jrb::adapters::compute::vulkan

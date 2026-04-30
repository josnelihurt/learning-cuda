#pragma once

#include <cstdint>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <vulkan/vulkan.hpp>
#pragma GCC diagnostic pop

namespace jrb::infrastructure::vulkan {

// Meyers singleton that owns the Vulkan instance, physical device, logical device,
// compute queue, and command pool. Mirrors the OpenCLContext singleton pattern.
// If no Vulkan-capable compute device is found, available() returns false and
// callers degrade gracefully.
class VulkanContext {
 public:
  static VulkanContext& GetInstance();

  bool available() const;
  const char* error_message() const;

  vk::Device device() const;
  vk::PhysicalDevice physical_device() const;
  vk::Queue queue() const;
  vk::CommandPool command_pool() const;
  uint32_t compute_queue_family_index() const;

 private:
  VulkanContext();
  ~VulkanContext();

  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  bool SetError(const char* message);

  bool available_;
  const char* error_message_;

  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::Queue queue_;
  vk::CommandPool command_pool_;
  uint32_t compute_queue_family_index_;
};

}  // namespace jrb::infrastructure::vulkan

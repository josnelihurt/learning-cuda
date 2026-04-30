#include "src/cpp_accelerator/adapters/compute/vulkan/context/context.h"

#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::adapters::compute::vulkan {

Context& Context::GetInstance() {
  static Context instance;
  return instance;
}

Context::Context()
    : available_(false),
      error_message_("not initialized"),
      compute_queue_family_index_(0xFFFFFFFFu) {
  vk::ApplicationInfo app_info("cpp-accelerator", VK_MAKE_VERSION(1, 0, 0), "No Engine",
                               VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_2);

  vk::InstanceCreateInfo instance_ci({}, &app_info);
  try {
    instance_ = vk::createInstance(instance_ci);
  } catch (const vk::SystemError&) {
    SetError("vk::createInstance failed — Vulkan not available");
    return;
  }

  std::vector<vk::PhysicalDevice> devices;
  try {
    devices = instance_.enumeratePhysicalDevices();
  } catch (const vk::SystemError&) {
    SetError("enumeratePhysicalDevices failed");
    return;
  }

  if (devices.empty()) {
    SetError("no Vulkan physical devices found");
    return;
  }

  // Pick the first device that exposes a compute queue family.
  for (auto& pd : devices) {
    auto families = pd.getQueueFamilyProperties();
    for (uint32_t i = 0; i < static_cast<uint32_t>(families.size()); ++i) {
      if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
        physical_device_ = pd;
        compute_queue_family_index_ = i;
        break;
      }
    }
    if (compute_queue_family_index_ != 0xFFFFFFFFu) break;
  }

  if (compute_queue_family_index_ == 0xFFFFFFFFu) {
    SetError("no Vulkan device with compute queue family found");
    return;
  }

  float priority = 1.0F;
  vk::DeviceQueueCreateInfo queue_ci({}, compute_queue_family_index_, 1, &priority);
  vk::PhysicalDeviceFeatures features{};
  vk::DeviceCreateInfo device_ci({}, 1, &queue_ci, 0, nullptr, 0, nullptr, &features);
  try {
    device_ = physical_device_.createDevice(device_ci);
  } catch (const vk::SystemError&) {
    SetError("failed to create Vulkan logical device");
    return;
  }

  queue_ = device_.getQueue(compute_queue_family_index_, 0);

  vk::CommandPoolCreateInfo pool_ci(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                    compute_queue_family_index_);
  try {
    command_pool_ = device_.createCommandPool(pool_ci);
  } catch (const vk::SystemError&) {
    SetError("failed to create Vulkan command pool");
    return;
  }

  available_ = true;
  error_message_ = "ok";
  spdlog::info("[Context] Vulkan context ready");
}

Context::~Context() {
  if (device_) {
    if (command_pool_) {
      device_.destroyCommandPool(command_pool_);
      command_pool_ = nullptr;
    }
    device_.destroy();
    device_ = nullptr;
  }
  if (instance_) {
    instance_.destroy();
    instance_ = nullptr;
  }
}

bool Context::SetError(const char* message) {
  error_message_ = message;
  spdlog::warn("[Context] {}", message);
  return false;
}

bool Context::available() const { return available_; }
const char* Context::error_message() const { return error_message_; }
vk::Device Context::device() const { return device_; }
vk::PhysicalDevice Context::physical_device() const { return physical_device_; }
vk::Queue Context::queue() const { return queue_; }
vk::CommandPool Context::command_pool() const { return command_pool_; }
uint32_t Context::compute_queue_family_index() const { return compute_queue_family_index_; }

}  // namespace jrb::adapters::compute::vulkan

#include "src/cpp_accelerator/adapters/compute/opencl/context/context.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#include <string_view>
#pragma GCC diagnostic pop

namespace jrb::adapters::compute::opencl {
namespace {
constexpr std::string_view kLogPrefix = "[OpenCLContext]";
}

Context& Context::GetInstance() {
  static Context instance;
  return instance;
}

Context::Context()
    : available_(false),
      error_message_("not initialized"),
      platform_(nullptr),
      device_(nullptr),
      context_(nullptr),
      queue_(nullptr) {
  cl_int err = clGetPlatformIDs(1, &platform_, nullptr);
  if (err != CL_SUCCESS || !platform_) {
    error_message_ = "clGetPlatformIDs failed — no OpenCL platform found";
    spdlog::warn("{} {}", kLogPrefix, error_message_);
    return;
  }

  // Prefer GPU; fall back to any device type.
  err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
  if (err != CL_SUCCESS || !device_) {
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_DEFAULT, 1, &device_, nullptr);
  }
  if (err != CL_SUCCESS || !device_) {
    error_message_ = "clGetDeviceIDs failed — no OpenCL device found";
    spdlog::warn("{} {}", kLogPrefix, error_message_);
    return;
  }

  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  if (err != CL_SUCCESS || !context_) {
    error_message_ = "clCreateContext failed";
    spdlog::warn("{} {}", kLogPrefix, error_message_);
    return;
  }

  queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
  if (err != CL_SUCCESS || !queue_) {
    clReleaseContext(context_);
    context_ = nullptr;
    error_message_ = "clCreateCommandQueueWithProperties failed";
    spdlog::warn("{} {}", kLogPrefix, error_message_);
    return;
  }

  available_ = true;
  error_message_ = "ok";
  spdlog::info("{} OpenCL context ready", kLogPrefix);
}

Context::~Context() {
  if (queue_)
    clReleaseCommandQueue(queue_);
  if (context_)
    clReleaseContext(context_);
}

bool Context::available() const {
  return available_;
}

const char* Context::error_message() const {
  return error_message_;
}

cl_context Context::context() const {
  return context_;
}

cl_device_id Context::device() const {
  return device_;
}

cl_command_queue Context::queue() const {
  return queue_;
}

}  // namespace jrb::adapters::compute::opencl

#pragma once

#include <CL/cl.h>

namespace jrb::adapters::compute::opencl {

// Meyer singleton owning the OpenCL platform / device / context / queue.
// Constructed on first use; the `available()` guard lets callers fall back
// gracefully when the host has no OpenCL-capable device.
class Context {
 public:
  static Context& GetInstance();

  bool available() const;
  const char* error_message() const;

  cl_context context() const;
  cl_device_id device() const;
  cl_command_queue queue() const;

 private:
  Context();
  ~Context();

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  bool available_;
  const char* error_message_;

  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_command_queue queue_;
};

}  // namespace jrb::adapters::compute::opencl

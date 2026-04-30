#pragma once

#include <CL/cl.h>

namespace jrb::infrastructure::opencl {

// Meyer singleton owning the OpenCL platform / device / context / queue.
// Constructed on first use; the `available()` guard lets callers fall back
// gracefully when the host has no OpenCL-capable device.
class OpenCLContext {
 public:
  static OpenCLContext& GetInstance();

  bool available() const;
  const char* error_message() const;

  cl_context context() const;
  cl_device_id device() const;
  cl_command_queue queue() const;

 private:
  OpenCLContext();
  ~OpenCLContext();

  OpenCLContext(const OpenCLContext&) = delete;
  OpenCLContext& operator=(const OpenCLContext&) = delete;

  bool available_;
  const char* error_message_;

  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_command_queue queue_;
};

}  // namespace jrb::infrastructure::opencl

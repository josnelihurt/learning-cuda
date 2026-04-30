#include "src/cpp_accelerator/adapters/compute/opencl/opencl_grayscale_filter.h"

#include <cstddef>
#include <cstring>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/adapters/compute/opencl/grayscale_kernel_blob.h"
#include "src/cpp_accelerator/adapters/compute/opencl/opencl_context.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::infrastructure::opencl {

OpenCLGrayscaleFilter::OpenCLGrayscaleFilter()
    : kernel_ready_(false), program_(nullptr), kernel_(nullptr) {}

OpenCLGrayscaleFilter::~OpenCLGrayscaleFilter() {
  if (kernel_) clReleaseKernel(kernel_);
  if (program_) clReleaseProgram(program_);
}

bool OpenCLGrayscaleFilter::EnsureKernel() {
  if (kernel_ready_) return true;

  auto& ctx = OpenCLContext::GetInstance();
  if (!ctx.available()) {
    spdlog::error("[OpenCLGrayscale] context unavailable: {}", ctx.error_message());
    return false;
  }

  cl_int err = CL_SUCCESS;
  cl_device_id device = ctx.device();

  // Prefer SPIR-V IL path; fall back to CL source text.
  const void* il_ptr = grayscale_kernel_blob::spirv();
  const size_t il_size = grayscale_kernel_blob::spirv_size_bytes();
  if (il_ptr && il_size > 0 && (il_size % sizeof(cl_uint)) == 0) {
    program_ = clCreateProgramWithIL(ctx.context(), il_ptr, il_size, &err);
    if (err != CL_SUCCESS || !program_) program_ = nullptr;
  }

  if (!program_) {
    const char* src = grayscale_kernel_blob::cl_src();
    const size_t src_len = grayscale_kernel_blob::cl_src_size_bytes();
    program_ = clCreateProgramWithSource(ctx.context(), 1, &src, &src_len, &err);
    if (err != CL_SUCCESS || !program_) {
      spdlog::error("[OpenCLGrayscale] clCreateProgramWithSource failed ({})", err);
      return false;
    }
  }

  err = clBuildProgram(program_, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    if (log_size > 1) {
      std::string log(log_size, '\0');
      clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
      spdlog::error("[OpenCLGrayscale] build error: {}", log);
    }
    clReleaseProgram(program_);
    program_ = nullptr;
    return false;
  }

  kernel_ = clCreateKernel(program_, "grayscale_bt601", &err);
  if (err != CL_SUCCESS || !kernel_) {
    spdlog::error("[OpenCLGrayscale] clCreateKernel failed ({})", err);
    clReleaseProgram(program_);
    program_ = nullptr;
    return false;
  }

  kernel_ready_ = true;
  return true;
}

bool OpenCLGrayscaleFilter::Apply(jrb::domain::interfaces::FilterContext& context) {
  if (!EnsureKernel()) return false;

  auto& ctx = OpenCLContext::GetInstance();
  const int width = context.input.width;
  const int height = context.input.height;
  const int channels = context.input.channels;

  if (channels != 3) {
    spdlog::error("[OpenCLGrayscale] expected 3-channel input, got {}", channels);
    return false;
  }

  const size_t in_size = static_cast<size_t>(width) * height * channels;
  const size_t out_size = static_cast<size_t>(width) * height;

  cl_int err = CL_SUCCESS;
  cl_mem d_in = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               in_size, const_cast<unsigned char*>(context.input.data),
                               &err);
  if (err != CL_SUCCESS) {
    spdlog::error("[OpenCLGrayscale] clCreateBuffer (input) failed ({})", err);
    return false;
  }

  cl_mem d_out = clCreateBuffer(ctx.context(), CL_MEM_WRITE_ONLY, out_size, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_in);
    spdlog::error("[OpenCLGrayscale] clCreateBuffer (output) failed ({})", err);
    return false;
  }

  clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_in);
  clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_out);
  clSetKernelArg(kernel_, 2, sizeof(int), &width);
  clSetKernelArg(kernel_, 3, sizeof(int), &height);

  const size_t global_work[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
  err = clEnqueueNDRangeKernel(ctx.queue(), kernel_, 2, nullptr, global_work, nullptr, 0, nullptr,
                               nullptr);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    spdlog::error("[OpenCLGrayscale] clEnqueueNDRangeKernel failed ({})", err);
    return false;
  }

  err = clEnqueueReadBuffer(ctx.queue(), d_out, CL_TRUE, 0, out_size, context.output.data, 0,
                            nullptr, nullptr);
  clReleaseMemObject(d_in);
  clReleaseMemObject(d_out);

  if (err != CL_SUCCESS) {
    spdlog::error("[OpenCLGrayscale] clEnqueueReadBuffer failed ({})", err);
    return false;
  }

  return true;
}

jrb::domain::interfaces::FilterType OpenCLGrayscaleFilter::GetType() const {
  return jrb::domain::interfaces::FilterType::GRAYSCALE;
}

bool OpenCLGrayscaleFilter::IsInPlace() const {
  return false;
}

}  // namespace jrb::infrastructure::opencl

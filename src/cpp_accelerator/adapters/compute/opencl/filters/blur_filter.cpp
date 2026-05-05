#include "src/cpp_accelerator/adapters/compute/opencl/filters/blur_filter.h"

#include <cstddef>
#include <cstring>
#include <string_view>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/adapters/compute/opencl/context/context.h"
#include "src/cpp_accelerator/adapters/compute/opencl/kernels/cl_blur_blob.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::adapters::compute::opencl {
namespace {
constexpr std::string_view kLogPrefix = "[OpenCLBlur]";
}

GaussianBlurFilter::GaussianBlurFilter()
    : kernel_ready_(false), program_(nullptr), kernel_h_(nullptr), kernel_v_(nullptr) {}

GaussianBlurFilter::~GaussianBlurFilter() {
  if (kernel_h_)
    clReleaseKernel(kernel_h_);
  if (kernel_v_)
    clReleaseKernel(kernel_v_);
  if (program_)
    clReleaseProgram(program_);
}

bool GaussianBlurFilter::EnsureKernels() {
  if (kernel_ready_)
    return true;

  auto& ctx = Context::GetInstance();
  if (!ctx.available()) {
    spdlog::error("{} context unavailable: {}", kLogPrefix, ctx.error_message());
    return false;
  }

  cl_int err = CL_SUCCESS;
  cl_device_id device = ctx.device();

  // Prefer SPIR-V IL path; fall back to CL source text.
  const void* il_ptr = cl_blur_blob::spirv();
  const size_t il_size = cl_blur_blob::spirv_size_bytes();
  if (il_ptr && il_size > 0 && (il_size % sizeof(cl_uint)) == 0) {
    program_ = clCreateProgramWithIL(ctx.context(), il_ptr, il_size, &err);
    if (err != CL_SUCCESS || !program_)
      program_ = nullptr;
  }

  if (!program_) {
    const char* src = cl_blur_blob::cl_src();
    const size_t src_len = cl_blur_blob::cl_src_size_bytes();
    program_ = clCreateProgramWithSource(ctx.context(), 1, &src, &src_len, &err);
    if (err != CL_SUCCESS || !program_) {
      spdlog::error("{} clCreateProgramWithSource failed ({})", kLogPrefix, err);
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
      spdlog::error("{} build error: {}", kLogPrefix, log);
    }
    clReleaseProgram(program_);
    program_ = nullptr;
    return false;
  }

  kernel_h_ = clCreateKernel(program_, "gaussian_blur_h", &err);
  if (err != CL_SUCCESS || !kernel_h_) {
    spdlog::error("{} clCreateKernel (h) failed ({})", kLogPrefix, err);
    clReleaseProgram(program_);
    program_ = nullptr;
    return false;
  }

  kernel_v_ = clCreateKernel(program_, "gaussian_blur_v", &err);
  if (err != CL_SUCCESS || !kernel_v_) {
    spdlog::error("{} clCreateKernel (v) failed ({})", kLogPrefix, err);
    clReleaseKernel(kernel_h_);
    kernel_h_ = nullptr;
    clReleaseProgram(program_);
    program_ = nullptr;
    return false;
  }

  kernel_ready_ = true;
  return true;
}

bool GaussianBlurFilter::Apply(jrb::domain::interfaces::FilterContext& context) {
  if (!EnsureKernels())
    return false;

  auto& ctx = Context::GetInstance();
  const int width = context.input.width;
  const int height = context.input.height;
  const int channels = context.input.channels;

  const size_t pixel_bytes = static_cast<size_t>(width) * height * channels;

  cl_int err = CL_SUCCESS;

  cl_mem d_src = clCreateBuffer(ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pixel_bytes,
                                const_cast<unsigned char*>(context.input.data), &err);
  if (err != CL_SUCCESS) {
    spdlog::error("{} clCreateBuffer (src) failed ({})", kLogPrefix, err);
    return false;
  }

  // Temporary buffer for horizontal pass result.
  cl_mem d_tmp = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, pixel_bytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_src);
    spdlog::error("{} clCreateBuffer (tmp) failed ({})", kLogPrefix, err);
    return false;
  }

  cl_mem d_dst = clCreateBuffer(ctx.context(), CL_MEM_WRITE_ONLY, pixel_bytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_tmp);
    spdlog::error("{} clCreateBuffer (dst) failed ({})", kLogPrefix, err);
    return false;
  }

  const size_t global_work[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};

  // Horizontal pass.
  clSetKernelArg(kernel_h_, 0, sizeof(cl_mem), &d_src);
  clSetKernelArg(kernel_h_, 1, sizeof(cl_mem), &d_tmp);
  clSetKernelArg(kernel_h_, 2, sizeof(int), &width);
  clSetKernelArg(kernel_h_, 3, sizeof(int), &height);
  clSetKernelArg(kernel_h_, 4, sizeof(int), &channels);
  err = clEnqueueNDRangeKernel(ctx.queue(), kernel_h_, 2, nullptr, global_work, nullptr, 0, nullptr,
                               nullptr);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_tmp);
    clReleaseMemObject(d_dst);
    spdlog::error("{} horizontal pass enqueue failed ({})", kLogPrefix, err);
    return false;
  }

  // Vertical pass.
  clSetKernelArg(kernel_v_, 0, sizeof(cl_mem), &d_tmp);
  clSetKernelArg(kernel_v_, 1, sizeof(cl_mem), &d_dst);
  clSetKernelArg(kernel_v_, 2, sizeof(int), &width);
  clSetKernelArg(kernel_v_, 3, sizeof(int), &height);
  clSetKernelArg(kernel_v_, 4, sizeof(int), &channels);
  err = clEnqueueNDRangeKernel(ctx.queue(), kernel_v_, 2, nullptr, global_work, nullptr, 0, nullptr,
                               nullptr);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_tmp);
    clReleaseMemObject(d_dst);
    spdlog::error("{} vertical pass enqueue failed ({})", kLogPrefix, err);
    return false;
  }

  err = clEnqueueReadBuffer(ctx.queue(), d_dst, CL_TRUE, 0, pixel_bytes, context.output.data, 0,
                            nullptr, nullptr);

  clReleaseMemObject(d_src);
  clReleaseMemObject(d_tmp);
  clReleaseMemObject(d_dst);

  if (err != CL_SUCCESS) {
    spdlog::error("{} clEnqueueReadBuffer failed ({})", kLogPrefix, err);
    return false;
  }

  return true;
}

jrb::domain::interfaces::FilterType GaussianBlurFilter::GetType() const {
  return jrb::domain::interfaces::FilterType::BLUR;
}

bool GaussianBlurFilter::IsInPlace() const {
  return false;
}

}  // namespace jrb::adapters::compute::opencl

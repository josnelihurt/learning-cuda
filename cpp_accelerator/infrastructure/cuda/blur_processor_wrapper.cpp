#include "cpp_accelerator/infrastructure/cuda/blur_processor.h"

#include <cmath>
#include <string>
#include <vector>

#include "cpp_accelerator/core/telemetry.h"
#include "cpp_accelerator/infrastructure/cuda/blur_kernel.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::infrastructure::cuda {

constexpr float SQRT_2PI = 2.506628274631000242F;

CudaGaussianBlurFilter::CudaGaussianBlurFilter(int kernel_size, float sigma, BorderMode border_mode,
                                               bool separable)
    : kernel_size_(kernel_size), sigma_(sigma), border_mode_(border_mode), separable_(separable) {
  InitializeKernel();
}

CudaGaussianBlurFilter::~CudaGaussianBlurFilter() = default;

void CudaGaussianBlurFilter::InitializeKernel() {
  if (kernel_size_ % 2 == 0) {
    kernel_size_++;
  }

  int radius = kernel_size_ / 2;
  kernel_.resize(kernel_size_);

  float sum = 0.0F;
  for (int i = 0; i < kernel_size_; i++) {
    float x = static_cast<float>(i - radius);
    float value = std::exp(-0.5F * (x * x) / (sigma_ * sigma_)) / (sigma_ * SQRT_2PI);
    kernel_[i] = value;
    sum += value;
  }

  for (int i = 0; i < kernel_size_; i++) {
    kernel_[i] /= sum;
  }
}

void CudaGaussianBlurFilter::SetKernelSize(int size) {
  kernel_size_ = size;
  InitializeKernel();
}

void CudaGaussianBlurFilter::SetSigma(float sigma) {
  sigma_ = sigma;
  InitializeKernel();
}

void CudaGaussianBlurFilter::SetBorderMode(BorderMode mode) {
  border_mode_ = mode;
}

FilterType CudaGaussianBlurFilter::GetType() const {
  return FilterType::BLUR;
}

bool CudaGaussianBlurFilter::IsInPlace() const {
  return false;
}

int CudaGaussianBlurFilter::GetKernelSize() const {
  return kernel_size_;
}

float CudaGaussianBlurFilter::GetSigma() const {
  return sigma_;
}

BorderMode CudaGaussianBlurFilter::GetBorderMode() const {
  return border_mode_;
}

bool CudaGaussianBlurFilter::Apply(FilterContext& context) {
  auto& telemetry = core::telemetry::TelemetryManager::GetInstance();
  auto span = telemetry.CreateSpan("cuda-blur", "apply_gaussian_blur_cuda");
  core::telemetry::ScopedSpan scoped_span(span);

  scoped_span.SetAttribute("image.width", static_cast<int64_t>(context.input.width));
  scoped_span.SetAttribute("image.height", static_cast<int64_t>(context.input.height));
  scoped_span.SetAttribute("image.channels", static_cast<int64_t>(context.input.channels));
  scoped_span.SetAttribute("kernel.size", static_cast<int64_t>(kernel_size_));
  scoped_span.SetAttribute("kernel.sigma", static_cast<double>(sigma_));
  scoped_span.SetAttribute("kernel.separable", separable_);

  if (!separable_) {
    scoped_span.RecordError("Non-separable blur not yet implemented in CUDA");
    spdlog::warn("Non-separable blur not yet implemented in CUDA");
    return false;
  }

  scoped_span.AddEvent("Calling pure CUDA blur kernel");

  cudaError_t error = cuda_apply_gaussian_blur_separable(
      context.input.data, context.output.data, context.input.width, context.input.height,
      context.input.channels, kernel_.data(), kernel_size_, static_cast<int>(border_mode_));

  if (error != cudaSuccess) {
    std::string error_msg = std::string("CUDA blur kernel failed: ") + cudaGetErrorString(error);
    spdlog::error(error_msg);
    scoped_span.RecordError(error_msg);
    return false;
  }

  scoped_span.AddEvent("CUDA blur kernel completed successfully");
  return true;
}

}  // namespace jrb::infrastructure::cuda

#include "cpp_accelerator/infrastructure/cuda/grayscale_filter.h"

#include <spdlog/spdlog.h>

#include "cpp_accelerator/core/telemetry.h"
#include "cpp_accelerator/domain/interfaces/image_buffer.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_kernel.h"

namespace jrb::infrastructure::cuda {

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::FilterType;

GrayscaleFilter::GrayscaleFilter(GrayscaleAlgorithm algorithm) : algorithm_(algorithm) {}

void GrayscaleFilter::SetAlgorithm(GrayscaleAlgorithm algorithm) {
  algorithm_ = algorithm;
}

GrayscaleAlgorithm GrayscaleFilter::GetAlgorithm() const {
  return algorithm_;
}

FilterType GrayscaleFilter::GetType() const {
  return FilterType::GRAYSCALE;
}

bool GrayscaleFilter::IsInPlace() const {
  return false;
}

bool GrayscaleFilter::Apply(FilterContext& context) {
  auto& telemetry = core::telemetry::TelemetryManager::GetInstance();
  auto span = telemetry.CreateSpan("cuda-grayscale", "apply_grayscale_cuda");
  core::telemetry::ScopedSpan scoped_span(span);

  if (!context.input.IsValid() || !context.output.IsValid()) {
    return false;
  }

  scoped_span.SetAttribute("image.width", static_cast<int64_t>(context.input.width));
  scoped_span.SetAttribute("image.height", static_cast<int64_t>(context.input.height));
  scoped_span.SetAttribute("image.channels", static_cast<int64_t>(context.input.channels));
  scoped_span.SetAttribute("algorithm", static_cast<int64_t>(algorithm_));

  // Accept 3 channels (RGB), 4 channels (RGBA), or 1 channel (grayscale)
  if (context.input.channels != 3 && context.input.channels != 4 && context.input.channels != 1) {
    std::string error_msg = "Unsupported channel count: " + std::to_string(context.input.channels);
    spdlog::error(error_msg);
    scoped_span.RecordError(error_msg);
    return false;
  }

  if (context.output.channels != 1) {
    std::string error_msg =
        "Output must have 1 channel for grayscale, got: " + std::to_string(context.output.channels);
    spdlog::error(error_msg);
    scoped_span.RecordError(error_msg);
    return false;
  }

  int algorithm_int = static_cast<int>(algorithm_);

  scoped_span.AddEvent("Calling pure CUDA grayscale kernel");

  cudaError_t error =
      cuda_convert_to_grayscale(context.input.data, context.output.data, context.input.width,
                                context.input.height, context.input.channels, algorithm_int);

  if (error != cudaSuccess) {
    std::string error_msg =
        std::string("CUDA grayscale kernel failed: ") + cudaGetErrorString(error);
    spdlog::error(error_msg);
    scoped_span.RecordError(error_msg);
    return false;
  }

  scoped_span.AddEvent("CUDA grayscale kernel completed successfully");
  return true;
}

}  // namespace jrb::infrastructure::cuda

#include <spdlog/spdlog.h>
#include <memory>

#include "cpp_accelerator/core/telemetry.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_kernel.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_processor.h"

namespace jrb::infrastructure::cuda {

GrayscaleProcessor::GrayscaleProcessor(GrayscaleAlgorithm algorithm) : algorithm_(algorithm) {}

void GrayscaleProcessor::set_algorithm(GrayscaleAlgorithm algorithm) {
  algorithm_ = algorithm;
}

void GrayscaleProcessor::convert_to_grayscale_cuda(const unsigned char* input,
                                                   unsigned char* output, int width, int height,
                                                   int channels) {
  auto& telemetry = core::telemetry::TelemetryManager::GetInstance();
  auto span = telemetry.CreateSpan("cuda-grayscale", "convert_to_grayscale_cuda");
  core::telemetry::ScopedSpan scoped_span(span);

  size_t input_size = width * height * channels;
  size_t output_size = width * height;

  scoped_span.SetAttribute("image.width", static_cast<int64_t>(width));
  scoped_span.SetAttribute("image.height", static_cast<int64_t>(height));
  scoped_span.SetAttribute("image.channels", static_cast<int64_t>(channels));
  scoped_span.SetAttribute("input.size_bytes", static_cast<int64_t>(input_size));
  scoped_span.SetAttribute("output.size_bytes", static_cast<int64_t>(output_size));

  scoped_span.AddEvent("Calling pure CUDA kernel");

  // Call the pure CUDA function
  cudaError_t error = cuda_convert_to_grayscale(input, output, width, height, channels,
                                                static_cast<int>(algorithm_));

  if (error != cudaSuccess) {
    std::string error_msg = std::string("CUDA kernel failed: ") + cudaGetErrorString(error);
    spdlog::error(error_msg);
    scoped_span.RecordError(error_msg);
    return;
  }

  scoped_span.SetAttribute("cuda.block_size.x", static_cast<int64_t>(16));
  scoped_span.SetAttribute("cuda.block_size.y", static_cast<int64_t>(16));

  scoped_span.AddEvent("CUDA kernel completed successfully");
}

bool GrayscaleProcessor::process(domain::interfaces::IImageSource& source,
                                 domain::interfaces::IImageSink& sink,
                                 const std::string& output_path) {
  if (!source.is_valid()) {
    spdlog::error("Invalid image source");
    return false;
  }

  int width = source.width();
  int height = source.height();
  int channels = source.channels();
  const unsigned char* input_data = source.data();

  spdlog::info("Processing {}x{} image (CUDA)", width, height);

  std::unique_ptr<unsigned char[]> output_data(new unsigned char[width * height]);
  convert_to_grayscale_cuda(input_data, output_data.get(), width, height, channels);
  return sink.write(output_path.c_str(), output_data.get(), width, height, 1);
}

}  // namespace jrb::infrastructure::cuda

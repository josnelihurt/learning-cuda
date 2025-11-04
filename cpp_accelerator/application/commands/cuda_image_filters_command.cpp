#include "cpp_accelerator/application/commands/cuda_image_filters_command.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::application::commands {

CudaImageFiltersCommand::CudaImageFiltersCommand(
    std::unique_ptr<domain::interfaces::IImageProcessor> processor,
    std::unique_ptr<domain::interfaces::IImageSource> source,
    std::unique_ptr<domain::interfaces::IImageSink> sink, std::string output_path)
    : processor_(std::move(processor)),
      source_(std::move(source)),
      sink_(std::move(sink)),
      output_path_(std::move(output_path)) {}

core::Result<void> CudaImageFiltersCommand::execute() {
  spdlog::info("Running CUDA image filters, output: {}", output_path_);

  if (!source_->is_valid()) {
    spdlog::error("Failed to load input image");
    return core::Result<void>::error("Failed to load input image", 1);
  }

  // Process image with GPU-accelerated filters
  bool success = processor_->process(*source_, *sink_, output_path_);

  if (!success) {
    spdlog::error("Image processing failed");
    return core::Result<void>::error("Image processing failed", 1);
  }

  spdlog::info("CUDA image filters completed successfully!");
  return core::Result<void>::ok("Image processing completed", 0);
}

}  // namespace jrb::application::commands

#include "cpp_accelerator/application/commands/cpu_image_filters_command.h"

#include <spdlog/spdlog.h>

namespace jrb::application::commands {

CpuImageFiltersCommand::CpuImageFiltersCommand(
    std::unique_ptr<domain::interfaces::IImageProcessor> processor,
    std::unique_ptr<domain::interfaces::IImageSource> source,
    std::unique_ptr<domain::interfaces::IImageSink> sink, std::string output_path)
    : processor_(std::move(processor)),
      source_(std::move(source)),
      sink_(std::move(sink)),
      output_path_(std::move(output_path)) {
}

core::Result<void> CpuImageFiltersCommand::execute() {
    spdlog::info("Running CPU image filters, output: {}", output_path_);

    if (!source_->is_valid()) {
        spdlog::error("Failed to load input image");
        return core::Result<void>::error("Failed to load input image", 1);
    }

    // Process image with CPU-based filters
    bool success = processor_->process(*source_, *sink_, output_path_);

    if (!success) {
        spdlog::error("Image processing failed");
        return core::Result<void>::error("Image processing failed", 1);
    }

    spdlog::info("CPU image filters completed successfully!");
    return core::Result<void>::ok("Image processing completed", 0);
}

}  // namespace jrb::application::commands

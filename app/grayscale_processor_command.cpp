#include "app/grayscale_processor_command.h"
#include "lib/image/image_loader.h"
#include "lib/image/image_writer.h"
#include "lib/cuda/grayscale_processor.h"
#include <spdlog/spdlog.h>

namespace jrb::app {

GrayscaleProcessorCommand::GrayscaleProcessorCommand(std::string input_path, std::string output_path)
    : input_path_(std::move(input_path))
    , output_path_(std::move(output_path)) {}

core::Result<void> GrayscaleProcessorCommand::execute() {
    spdlog::info("Running grayscale image processing...");
    spdlog::info("Input: {}", input_path_);
    spdlog::info("Output: {}", output_path_);
    
    jrb::lib::image::ImageLoader loader(input_path_.c_str());
    
    if (!loader.is_loaded()) {
        spdlog::error("Failed to load input image");
        return core::Result<void>::error("Failed to load input image", 1);
    }
    
    jrb::lib::image::ImageWriter writer;
    jrb::lib::cuda::GrayscaleProcessor processor;
    
    bool success = processor.process(loader, writer, output_path_.c_str());
    
    if (success) {
        spdlog::info("Image processing completed successfully!");
        return core::Result<void>::ok("Image processing completed", 0);
    } else {
        spdlog::error("Image processing failed");
        return core::Result<void>::error("Image processing failed", 1);
    }
}

}  // namespace jrb::app


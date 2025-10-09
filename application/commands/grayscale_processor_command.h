#pragma once

#include "application/commands/command_interface.h"
#include "domain/interfaces/processors/i_image_processor.h"
#include "domain/interfaces/image_source.h"
#include "domain/interfaces/image_sink.h"
#include <string>
#include <memory>

namespace jrb::application::commands {

class GrayscaleProcessorCommand final: public ICommand {
public:
    GrayscaleProcessorCommand(
        std::unique_ptr<domain::interfaces::IImageProcessor> processor,
        std::unique_ptr<domain::interfaces::IImageSource> source,
        std::unique_ptr<domain::interfaces::IImageSink> sink,
        std::string output_path
    );
    ~GrayscaleProcessorCommand() override = default;
    
    core::Result<void> execute() override;

private:
    std::unique_ptr<domain::interfaces::IImageProcessor> processor_;
    std::unique_ptr<domain::interfaces::IImageSource> source_;
    std::unique_ptr<domain::interfaces::IImageSink> sink_;
    std::string output_path_;
};

}  // namespace jrb::application::commands


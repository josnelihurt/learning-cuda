#pragma once

#include <memory>
#include <string>

#include "cpp_accelerator/application/commands/command_interface.h"
#include "cpp_accelerator/domain/interfaces/image_sink.h"
#include "cpp_accelerator/domain/interfaces/image_source.h"
#include "cpp_accelerator/domain/interfaces/processors/i_image_processor.h"

namespace jrb::application::commands {

class CpuImageFiltersCommand final : public ICommand {
   public:
    CpuImageFiltersCommand(std::unique_ptr<domain::interfaces::IImageProcessor> processor,
                           std::unique_ptr<domain::interfaces::IImageSource> source,
                           std::unique_ptr<domain::interfaces::IImageSink> sink,
                           std::string output_path);
    ~CpuImageFiltersCommand() override = default;

    core::Result<void> execute() override;

   private:
    std::unique_ptr<domain::interfaces::IImageProcessor> processor_;
    std::unique_ptr<domain::interfaces::IImageSource> source_;
    std::unique_ptr<domain::interfaces::IImageSink> sink_;
    std::string output_path_;
};

}  // namespace jrb::application::commands

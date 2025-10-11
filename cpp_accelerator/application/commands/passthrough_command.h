#pragma once

#include <memory>

#include "cpp_accelerator/application/commands/command_interface.h"
#include "cpp_accelerator/domain/interfaces/processors/i_image_processor.h"

namespace jrb::application::commands {

class PassthroughCommand final : public ICommand {
   public:
    explicit PassthroughCommand(std::unique_ptr<domain::interfaces::IImageProcessor> processor);
    ~PassthroughCommand() override = default;

    core::Result<void> execute() override;

   private:
    std::unique_ptr<domain::interfaces::IImageProcessor> processor_;
};

}  // namespace jrb::application::commands

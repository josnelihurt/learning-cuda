#pragma once

#include "cpp_accelerator/application/commands/command_interface.h"
#include "cpp_accelerator/domain/interfaces/processors/i_image_processor.h"
#include <memory>

namespace jrb::application::commands {

class SimpleKernelCommand final: public ICommand {
public:
    explicit SimpleKernelCommand(std::unique_ptr<domain::interfaces::IImageProcessor> processor);
    ~SimpleKernelCommand() override = default;
    
    core::Result<void> execute() override;

private:
    std::unique_ptr<domain::interfaces::IImageProcessor> processor_;
};

}  // namespace jrb::application::commands


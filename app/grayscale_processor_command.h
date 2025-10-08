#pragma once

#include "app/command_interface.h"
#include <string>

namespace jrb::app {

class GrayscaleProcessorCommand : public ICommand {
public:
    GrayscaleProcessorCommand(std::string input_path, std::string output_path);
    ~GrayscaleProcessorCommand() override = default;
    
    core::Result<void> execute() override;

private:
    std::string input_path_;
    std::string output_path_;
};

}  // namespace jrb::app


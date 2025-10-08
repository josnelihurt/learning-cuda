#pragma once

#include "app/command_interface.h"

namespace jrb::app {

class SimpleKernelCommand : public ICommand {
public:
    SimpleKernelCommand() = default;
    ~SimpleKernelCommand() override = default;
    
    core::Result<void> execute() override;
};

}  // namespace jrb::app


#pragma once

#include "core/result.h"

namespace jrb::application::commands {

class ICommand {
public:
    virtual ~ICommand() = default;
    virtual core::Result<void> execute() = 0;
};

}  // namespace jrb::application::commands


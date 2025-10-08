#pragma once

#include "core/result.h"

namespace jrb::app {

class ICommand {
public:
    virtual ~ICommand() = default;
    virtual core::Result<void> execute() = 0;
};

}  // namespace jrb::app


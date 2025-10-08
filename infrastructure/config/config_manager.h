#pragma once

#include "infrastructure/config/models/program_config.h"
#include "core/result.h"
#include <string>
#include <span>

namespace jrb::infrastructure::config {

class ConfigManager {
public:
    // Parse command line arguments and return configuration
    static core::Result<models::ProgramConfig> parse(std::span<const char*> args);

private:
    ConfigManager() = delete;
};

}  // namespace jrb::infrastructure::config

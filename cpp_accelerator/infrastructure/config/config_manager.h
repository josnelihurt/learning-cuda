#pragma once

#include "cpp_accelerator/infrastructure/config/models/program_config.h"
#include "cpp_accelerator/core/result.h"
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

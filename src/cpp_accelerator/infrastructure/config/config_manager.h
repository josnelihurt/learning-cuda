#pragma once

#include <span>
#include <string>
#include "src/cpp_accelerator/core/result.h"
#include "src/cpp_accelerator/infrastructure/config/models/program_config.h"

namespace jrb::infrastructure::config {

class ConfigManager {
public:
  // Parse command line arguments and return configuration
  static core::Result<models::ProgramConfig> parse(std::span<const char*> args);

private:
  ConfigManager() = delete;
};

}  // namespace jrb::infrastructure::config

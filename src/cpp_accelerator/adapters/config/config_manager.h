#pragma once

#include <span>
#include <string>
#include "src/cpp_accelerator/core/result.h"
#include "src/cpp_accelerator/adapters/config/models/program_config.h"

namespace jrb::adapters::config {

class ConfigManager {
public:
  // Parse command line arguments and return configuration
  static core::Result<models::ProgramConfig> parse(std::span<const char*> args);

private:
  ConfigManager() = delete;
};

}  // namespace jrb::adapters::config

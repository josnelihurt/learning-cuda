#include <span>
#include "cpp_accelerator/application/commands/command_factory.h"
#include "cpp_accelerator/core/logger.h"
#include "cpp_accelerator/infrastructure/config/config_manager.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb {
int main(int argc, const char** argv) {
  core::initialize_logger();

  std::span<const char*> args(argv, argc);
  auto config_result = infrastructure::config::ConfigManager::parse(args);

  if (!config_result) {
    if (config_result.exit_code != 0) {
      spdlog::error("Configuration error: {}", config_result.message);
    }
    return config_result.exit_code;
  }

  const auto& config = config_result.value.value();

  application::commands::CommandFactory factory;
  auto command = factory.create(config.program_type, config);

  if (!command) {
    spdlog::error("Unknown program type");
    return 1;
  }

  auto result = command->execute();
  if (!result) {
    spdlog::error("Command execution failed: {}", result.message);
  }

  return result.exit_code;

}  // namespace jrb

}  // namespace jrb

// entry point for bazel
int main(int argc, const char** argv) {
  return jrb::main(argc, argv);
}
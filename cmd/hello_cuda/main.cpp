#include "core/logger.h"
#include "infrastructure/config/config_manager.h"
#include "app/command_factory.h"
#include <spdlog/spdlog.h>
#include <span>

using namespace jrb;

int main(int argc, const char** argv) {
    // Initialize logger
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
    
    app::CommandFactory factory;
    auto command = factory.create(config.program_type, config);
    
    if (!command) {
        spdlog::error("Unknown program type");
        return 1;
    }
    
    // Execute command
    auto result = command->execute();
    if (!result) {
        spdlog::error("Command execution failed: {}", result.message);
    }
    
    return result.exit_code;
}
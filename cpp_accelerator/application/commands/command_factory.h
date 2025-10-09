#pragma once

#include "cpp_accelerator/application/commands/command_interface.h"
#include "cpp_accelerator/infrastructure/config/models/program_config.h"
#include <memory>
#include <functional>
#include <map>

namespace jrb::application::commands {

class CommandFactory {
public:
    using CommandCreator = std::function<std::unique_ptr<ICommand>(
        const infrastructure::config::models::ProgramConfig&
    )>;
    
    CommandFactory();
    
    // Create a command based on program type and config
    std::unique_ptr<ICommand> create(
        infrastructure::config::models::ProgramType type,
        const infrastructure::config::models::ProgramConfig& config
    ) const;
    
    // Check if a command type is registered
    bool is_registered(infrastructure::config::models::ProgramType type) const;

private:
    void register_commands();
    
    std::map<
        infrastructure::config::models::ProgramType, 
        CommandCreator
    > creators_;
};

}  // namespace jrb::application::commands

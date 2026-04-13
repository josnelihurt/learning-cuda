#include "cpp_accelerator/application/commands/command_factory.h"

namespace jrb::application::commands {

CommandFactory::CommandFactory() {
  register_commands();
}

void CommandFactory::register_commands() {
  // Command factory is currently empty as all commands have been migrated to FilterPipeline.
  // This factory is kept as a placeholder for future command implementations if needed.
  // Production code uses FilterPipeline directly via ports/cgo and ports/shared_lib.
}

std::unique_ptr<ICommand> CommandFactory::create(
    infrastructure::config::models::ProgramType type,
    const infrastructure::config::models::ProgramConfig& config) const {
  auto it = creators_.find(type);
  if (it == creators_.end()) {
    return nullptr;
  }
  return it->second(config);
}

bool CommandFactory::is_registered(infrastructure::config::models::ProgramType type) const {
  return creators_.find(type) != creators_.end();
}

}  // namespace jrb::application::commands

#include "app/command_factory.h"
#include "app/simple_kernel_command.h"
#include "app/grayscale_processor_command.h"

namespace jrb::app {

CommandFactory::CommandFactory() {
    register_commands();
}

void CommandFactory::register_commands() {
    // Register Simple Kernel Command
    creators_[infrastructure::config::models::ProgramType::Simple] = 
        [](const infrastructure::config::models::ProgramConfig& config) {
            return std::make_unique<SimpleKernelCommand>();
        };
    
    // Register Grayscale Processor Command
    creators_[infrastructure::config::models::ProgramType::Grayscale] = 
        [](const infrastructure::config::models::ProgramConfig& config) {
            return std::make_unique<GrayscaleProcessorCommand>(
                config.input_image_path,
                config.output_image_path
            );
        };
}

std::unique_ptr<ICommand> CommandFactory::create(
    infrastructure::config::models::ProgramType type,
    const infrastructure::config::models::ProgramConfig& config
) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) {
        return nullptr;
    }
    return it->second(config);
}

bool CommandFactory::is_registered(
    infrastructure::config::models::ProgramType type
) const {
    return creators_.find(type) != creators_.end();
}

}  // namespace jrb::app

#include "cpp_accelerator/application/commands/command_factory.h"
#include "cpp_accelerator/application/commands/simple_kernel_command.h"
#include "cpp_accelerator/application/commands/grayscale_processor_command.h"
#include "cpp_accelerator/infrastructure/cuda/simple_kernel_processor.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_processor.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"
#include "cpp_accelerator/infrastructure/image/image_writer.h"

namespace jrb::application::commands {

CommandFactory::CommandFactory() {
    register_commands();
}

void CommandFactory::register_commands() {
    // Register Simple Kernel Command with DI
    creators_[infrastructure::config::models::ProgramType::Simple] = 
        [](const infrastructure::config::models::ProgramConfig& config) {
            auto processor = std::make_unique<infrastructure::cuda::SimpleKernelProcessor>();
            return std::make_unique<SimpleKernelCommand>(std::move(processor));
        };
    
    // Register Grayscale Processor Command with DI
    creators_[infrastructure::config::models::ProgramType::Grayscale] = 
        [](const infrastructure::config::models::ProgramConfig& config) {
            auto processor = std::make_unique<infrastructure::cuda::GrayscaleProcessor>();
            auto source = std::make_unique<infrastructure::image::ImageLoader>(
                config.input_image_path.c_str()
            );
            auto sink = std::make_unique<infrastructure::image::ImageWriter>();
            
            return std::make_unique<GrayscaleProcessorCommand>(
                std::move(processor),
                std::move(source),
                std::move(sink),
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

}  // namespace jrb::application::commands

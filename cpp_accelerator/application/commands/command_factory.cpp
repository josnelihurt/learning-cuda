#include "cpp_accelerator/application/commands/command_factory.h"

#include "cpp_accelerator/application/commands/cpu_image_filters_command.h"
#include "cpp_accelerator/application/commands/cuda_image_filters_command.h"
#include "cpp_accelerator/application/commands/passthrough_command.h"
#include "cpp_accelerator/infrastructure/cpu/grayscale_processor.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_processor.h"
#include "cpp_accelerator/infrastructure/cuda/simple_kernel_processor.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"
#include "cpp_accelerator/infrastructure/image/image_writer.h"

namespace jrb::application::commands {

CommandFactory::CommandFactory() {
    register_commands();
}

void CommandFactory::register_commands() {
    creators_[infrastructure::config::models::ProgramType::Passthrough] =
        [](const infrastructure::config::models::ProgramConfig& config) {
            auto processor = std::make_unique<infrastructure::cuda::SimpleKernelProcessor>();
            return std::make_unique<PassthroughCommand>(std::move(processor));
        };

    creators_[infrastructure::config::models::ProgramType::CudaImageFilters] =
        [](const infrastructure::config::models::ProgramConfig& config) {
            auto processor = std::make_unique<infrastructure::cuda::GrayscaleProcessor>();
            auto source = std::make_unique<infrastructure::image::ImageLoader>(
                config.input_image_path.c_str());
            auto sink = std::make_unique<infrastructure::image::ImageWriter>();
            return std::make_unique<CudaImageFiltersCommand>(
                std::move(processor), std::move(source), std::move(sink), config.output_image_path);
        };

    creators_[infrastructure::config::models::ProgramType::CpuImageFilters] =
        [](const infrastructure::config::models::ProgramConfig& config) {
            auto processor = std::make_unique<infrastructure::cpu::CpuGrayscaleProcessor>();
            auto source = std::make_unique<infrastructure::image::ImageLoader>(
                config.input_image_path.c_str());
            auto sink = std::make_unique<infrastructure::image::ImageWriter>();
            return std::make_unique<CpuImageFiltersCommand>(
                std::move(processor), std::move(source), std::move(sink), config.output_image_path);
        };
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

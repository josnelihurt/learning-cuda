#include "cpp_accelerator/application/commands/command_factory.h"

#include "cpp_accelerator/application/commands/passthrough_command.h"
#include "cpp_accelerator/infrastructure/cuda/simple_kernel_processor.h"

namespace jrb::application::commands {

CommandFactory::CommandFactory() {
  register_commands();
}

void CommandFactory::register_commands() {
  creators_[infrastructure::config::models::ProgramType::Passthrough] =
      []([[maybe_unused]] const infrastructure::config::models::ProgramConfig& config) {
        auto processor = std::make_unique<infrastructure::cuda::SimpleKernelProcessor>();
        return std::make_unique<PassthroughCommand>(std::move(processor));
      };

  // TODO: Migrate CudaImageFilters and CpuImageFilters commands to use FilterPipeline
  // These commands were temporarily disabled during processor removal.
  // They should be reimplemented using FilterPipeline with GrayscaleFilter.
  // creators_[infrastructure::config::models::ProgramType::CudaImageFilters] = ...
  // creators_[infrastructure::config::models::ProgramType::CpuImageFilters] = ...
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

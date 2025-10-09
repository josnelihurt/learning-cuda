#include "cpp_accelerator/application/commands/simple_kernel_command.h"
#include "cpp_accelerator/domain/interfaces/image_source.h"
#include "cpp_accelerator/domain/interfaces/image_sink.h"
#include <spdlog/spdlog.h>

namespace jrb::application::commands {

SimpleKernelCommand::SimpleKernelCommand(std::unique_ptr<domain::interfaces::IImageProcessor> processor)
    : processor_(std::move(processor)) {}

core::Result<void> SimpleKernelCommand::execute() {
    spdlog::info("Running simple CUDA kernel...");
    
    // Simple kernel doesn't need actual image source/sink, just run the processor
    // We'll pass dummy references (this will be handled inside the processor)
    class DummySource : public domain::interfaces::IImageSource {
        int width() const override { return 0; }
        int height() const override { return 0; }
        int channels() const override { return 0; }
        const unsigned char* data() const override { return nullptr; }
        bool is_valid() const override { return true; }
    };
    
    class DummySink : public domain::interfaces::IImageSink {
        bool write(const char*, const unsigned char*, int, int, int) override { return true; }
    };
    
    DummySource source;
    DummySink sink;
    
    processor_->process(source, sink, "");
    
    spdlog::info("Simple kernel completed successfully!");
    
    return core::Result<void>::ok("Simple kernel execution completed", 0);
}

}  // namespace jrb::application::commands


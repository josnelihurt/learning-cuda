#include "cpp_accelerator/application/commands/passthrough_command.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "cpp_accelerator/domain/interfaces/image_sink.h"
#include "cpp_accelerator/domain/interfaces/image_source.h"

namespace jrb::application::commands {

PassthroughCommand::PassthroughCommand(
    std::unique_ptr<domain::interfaces::IImageProcessor> processor)
    : processor_(std::move(processor)) {}

core::Result<void> PassthroughCommand::execute() {
  spdlog::info("Running passthrough/noop command...");

  // Passthrough command doesn't process anything, just returns success
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

  spdlog::info("Passthrough command completed successfully!");

  return core::Result<void>::ok("Passthrough command execution completed", 0);
}

}  // namespace jrb::application::commands

#include "cpp_accelerator/infrastructure/config/config_manager.h"
#include <spdlog/spdlog.h>
#include <lyra/lyra.hpp>
#include <sstream>

namespace jrb::infrastructure::config {

core::Result<models::ProgramConfig> ConfigManager::parse(std::span<const char*> args) {
  // Convert span to argc/argv for Lyra
  int argc = static_cast<int>(args.size());
  const char** argv = args.data();

  // Default values
  std::string input_path = "data/static_images/lena.png";
  std::string output_path = "data/output.png";
  std::string type_str = "grayscale";
  bool help = false;

  // Build CLI parser
  auto cli = lyra::cli() | lyra::help(help) |
             lyra::opt(input_path, "input")["-i"]["--input"](
                 "Path to input image (default: data/static_images/lena.png)") |
             lyra::opt(output_path, "output")["-o"]["--output"](
                 "Path to output image (default: data/output.png)") |
             lyra::opt(type_str, "type")["-t"]["--type"](
                 "Program type: 'simple' or 'grayscale' (default: grayscale)");

  // Parse arguments
  auto parse_result = cli.parse({argc, argv});

  if (!parse_result) {
    return core::Result<models::ProgramConfig>::error(parse_result.message());
  }

  if (help) {
    std::ostringstream oss;
    oss << cli;
    spdlog::info("{}", oss.str());
    return core::Result<models::ProgramConfig>::error("Help requested", 0);
  }

  // Parse program type
  models::ProgramType program_type;
  if (type_str == "simple") {
    program_type = models::ProgramType::Passthrough;
  } else if (type_str == "grayscale") {
    program_type = models::ProgramType::CudaImageFilters;
  } else {
    return core::Result<models::ProgramConfig>::error("Invalid program type: '" + type_str +
                                                      "'. Must be 'simple' or 'grayscale'.");
  }

  // Return configuration successfully
  models::ProgramConfig config{.input_image_path = input_path,
                               .output_image_path = output_path,
                               .program_type = program_type};

  return core::Result<models::ProgramConfig>::ok(std::move(config),
                                                 "Configuration parsed successfully");
}

}  // namespace jrb::infrastructure::config

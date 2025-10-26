#include "cpp_accelerator/core/logger.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace jrb::core {

void initialize_logger() {
  // Console sink for stdout (existing behavior)
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  // File sink for JSON structured logs
  auto file_sink =
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("/tmp/cppaccelerator.log", true);
  file_sink->set_pattern(
      "{\"timestamp\":\"%Y-%m-%dT%H:%M:%S.%eZ\",\"level\":\"%l\",\"message\":\"%v\",\"source\":"
      "\"cpp\"}");

  // Create logger with both sinks
  auto logger =
      std::make_shared<spdlog::logger>("main", spdlog::sinks_init_list{console_sink, file_sink});
  logger->set_level(spdlog::level::info);

  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::info);
}

}  // namespace jrb::core

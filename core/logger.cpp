#include "core/logger.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace jrb::core {

void initialize_logger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    auto logger = std::make_shared<spdlog::logger>("main", console_sink);
    logger->set_level(spdlog::level::debug);
    
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
}

}  // namespace jrb::core


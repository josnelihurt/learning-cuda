#include "cpp_accelerator/core/logger.h"

#include <fstream>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::core {

// Build-time constant to verify library reloading
constexpr const char* LOGGER_BUILD_TIMESTAMP = __DATE__ " " __TIME__;

void initialize_logger() {
  std::cout << "DEBUG: initialize_logger() called [BUILD: " << LOGGER_BUILD_TIMESTAMP << "]"
            << std::endl;

  const char* log_file = "/tmp/cppaccelerator.log";
  std::shared_ptr<spdlog::logger> logger;

  std::cout << "DEBUG: Attempting to create file logger at: " << log_file << std::endl;

  try {
    // Use basic_logger_mt which handles file creation and lifecycle better
    logger = spdlog::basic_logger_mt("cpp_accelerator", log_file, true);
    std::cout << "DEBUG: basic_logger_mt created successfully" << std::endl;

    // Set JSON pattern for structured logs with Unix epoch timestamp
    logger->set_pattern(
        "{\"timestamp\":%E,\"level\":\"%l\",\"message\":\"%v\",\"source\":"
        "\"cpp\"}\n");

    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::info);

    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);

    // Use stdout for initialization message to ensure visibility
    std::cout << "Logger initialized successfully [BUILD: " << LOGGER_BUILD_TIMESTAMP
              << "] [LIB_RELOAD_TEST]" << std::endl;

    // Verify file was created by attempting to read it
    std::ifstream verify(log_file);
    if (!verify.good()) {
      std::cerr << "WARNING: Logger file sink created but file may not be accessible: " << log_file
                << std::endl;
    }
    verify.close();

  } catch (const spdlog::spdlog_ex& ex) {
    std::cerr << "ERROR: spdlog initialization failed: " << ex.what() << std::endl;
    std::cerr << "ERROR: Logger build timestamp: " << LOGGER_BUILD_TIMESTAMP << std::endl;
    std::cerr << "ERROR: Attempting fallback to console logger..." << std::endl;

    try {
      // Fallback to console logger
      auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
      logger = std::make_shared<spdlog::logger>("cpp_accelerator",
                                                spdlog::sinks_init_list{console_sink});
      logger->set_level(spdlog::level::info);
      spdlog::set_default_logger(logger);
      spdlog::set_level(spdlog::level::info);

      std::cerr << "WARNING: Using console logger as fallback (file logger failed)" << std::endl;
      logger->error("Failed to initialize file logger, using console only");
      logger->flush();
    } catch (const std::exception& fallback_ex) {
      std::cerr << "FATAL: Both file and console logger initialization failed: "
                << fallback_ex.what() << std::endl;
      std::cerr << "FATAL: Logger build timestamp: " << LOGGER_BUILD_TIMESTAMP << std::endl;
    }
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: Unexpected exception during logger initialization: " << ex.what()
              << std::endl;
    std::cerr << "ERROR: Logger build timestamp: " << LOGGER_BUILD_TIMESTAMP << std::endl;
    std::cerr << "ERROR: Attempting fallback to console logger..." << std::endl;

    try {
      // Fallback to console logger
      auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
      logger = std::make_shared<spdlog::logger>("cpp_accelerator",
                                                spdlog::sinks_init_list{console_sink});
      logger->set_level(spdlog::level::info);
      spdlog::set_default_logger(logger);
      spdlog::set_level(spdlog::level::info);

      std::cerr << "WARNING: Using console logger as fallback (file logger failed)" << std::endl;
    } catch (const std::exception& fallback_ex) {
      std::cerr << "FATAL: Both file and console logger initialization failed: "
                << fallback_ex.what() << std::endl;
      std::cerr << "FATAL: Logger build timestamp: " << LOGGER_BUILD_TIMESTAMP << std::endl;
    }
  }
}

}  // namespace jrb::core

#include "cpp_accelerator/core/logger.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "cpp_accelerator/core/otel_log_sink.h"

namespace jrb::core {

// Build-time constant to verify library reloading
constexpr const char* LOGGER_BUILD_TIMESTAMP = __DATE__ " " __TIME__;

void initialize_logger(const std::string& otlp_endpoint, bool remote_enabled,
                       const std::string& environment) {
  // Read configuration from environment variables if not provided
  std::string endpoint = otlp_endpoint;
  bool enabled = remote_enabled;
  std::string env = environment;

  const char* env_endpoint = std::getenv("OTEL_LOGS_ENDPOINT");
  const char* env_enabled = std::getenv("OTEL_LOGS_ENABLED");
  const char* env_environment = std::getenv("OTEL_ENVIRONMENT");

  if (env_endpoint && env_endpoint[0] != '\0') {
    endpoint = env_endpoint;
  }
  if (env_enabled) {
    enabled = (std::string(env_enabled) == "true" || std::string(env_enabled) == "1");
  }
  if (env_environment && env_environment[0] != '\0') {
    env = env_environment;
  }

  std::cout << "DEBUG: initialize_logger() called [BUILD: " << LOGGER_BUILD_TIMESTAMP << "]"
            << std::endl;

  const char* log_file = "/tmp/cppaccelerator.log";
  std::shared_ptr<spdlog::logger> logger;
  std::vector<spdlog::sink_ptr> sinks;

  std::cout << "DEBUG: Attempting to create file logger at: " << log_file << std::endl;

  try {
    // Create file sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
    file_sink->set_pattern(
        "{\"timestamp\":%E,\"level\":\"%l\",\"message\":\"%v\",\"source\":"
        "\"cpp\"}\n");
    sinks.push_back(file_sink);
    std::cout << "DEBUG: basic_file_sink_mt created successfully" << std::endl;

    // Add OpenTelemetry sink if enabled
    if (enabled && !endpoint.empty()) {
      try {
        auto otel_sink = std::make_shared<OtelLogSink>(endpoint, env);
        sinks.push_back(otel_sink);
        std::cout << "DEBUG: OpenTelemetry log sink added (endpoint: " << endpoint << ")"
                  << std::endl;
      } catch (const std::exception& otel_ex) {
        std::cerr << "WARNING: Failed to create OpenTelemetry sink: " << otel_ex.what()
                  << std::endl;
        std::cerr << "WARNING: Continuing with file logger only" << std::endl;
      }
    } else if (enabled) {
      std::cerr << "WARNING: OpenTelemetry logging enabled but endpoint not configured"
                << std::endl;
    }

    // Create logger with all sinks
    logger = std::make_shared<spdlog::logger>("cpp_accelerator", sinks.begin(), sinks.end());

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
      logger->set_level(spdlog::level::debug);
      spdlog::set_default_logger(logger);
      spdlog::set_level(spdlog::level::debug);

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
      logger->set_level(spdlog::level::debug);
      spdlog::set_default_logger(logger);
      spdlog::set_level(spdlog::level::debug);

      std::cerr << "WARNING: Using console logger as fallback (file logger failed)" << std::endl;
    } catch (const std::exception& fallback_ex) {
      std::cerr << "FATAL: Both file and console logger initialization failed: "
                << fallback_ex.what() << std::endl;
      std::cerr << "FATAL: Logger build timestamp: " << LOGGER_BUILD_TIMESTAMP << std::endl;
    }
  }
}

}  // namespace jrb::core

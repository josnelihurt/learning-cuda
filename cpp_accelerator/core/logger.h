#pragma once

#include <string>

namespace jrb::core {

// Initialize the global logger with console output and colors
// Optional parameters for OpenTelemetry remote logging:
// - otlp_endpoint: OTLP HTTP endpoint URL (e.g., "https://otel-cuda-demo.josnelihurt.me/v1/logs")
// - remote_enabled: Enable remote logging via OpenTelemetry
// - environment: Environment name (e.g., "production", "development")
void initialize_logger(const std::string& otlp_endpoint = "", bool remote_enabled = false,
                       const std::string& environment = "development");

}  // namespace jrb::core

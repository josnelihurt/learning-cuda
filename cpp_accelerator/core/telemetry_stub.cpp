#include "cpp_accelerator/core/telemetry.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

// Stub implementation for when OpenTelemetry C++ dependencies are not available
// This allows the code to compile while we work on resolving OpenTelemetry C++ dependencies

namespace jrb::core::telemetry {

TelemetryManager& TelemetryManager::GetInstance() {
  static TelemetryManager instance;
  return instance;
}

bool TelemetryManager::Initialize([[maybe_unused]] const std::string& service_name,
                                  [[maybe_unused]] const std::string& collector_endpoint,
                                  bool enabled) {
  enabled_ = enabled;
  initialized_ = true;
  if (enabled_) {
    spdlog::info("Telemetry stub initialized (OpenTelemetry C++ not fully linked yet)");
  }
  return true;
}

void TelemetryManager::Shutdown() {
  if (initialized_) {
    spdlog::info("Telemetry stub shut down");
  }
}

TracerPtr TelemetryManager::GetTracer([[maybe_unused]] const std::string& name) {
  return nullptr;
}

SpanPtr TelemetryManager::CreateSpan([[maybe_unused]] const std::string& tracer_name,
                                     [[maybe_unused]] const std::string& span_name) {
  return std::shared_ptr<StubSpan>(new StubSpan());
}

SpanPtr TelemetryManager::CreateChildSpan([[maybe_unused]] const std::string& tracer_name,
                                          [[maybe_unused]] const std::string& span_name,
                                          [[maybe_unused]] const std::string& parent_trace_id,
                                          [[maybe_unused]] const std::string& parent_span_id,
                                          [[maybe_unused]] uint8_t trace_flags) {
  return std::shared_ptr<StubSpan>(new StubSpan());
}

void ScopedSpan::SetAttribute([[maybe_unused]] const std::string& key,
                              [[maybe_unused]] const std::string& value) {
  // Stub
}

void ScopedSpan::SetAttribute([[maybe_unused]] const std::string& key,
                              [[maybe_unused]] int64_t value) {
  // Stub
}

void ScopedSpan::SetAttribute([[maybe_unused]] const std::string& key,
                              [[maybe_unused]] double value) {
  // Stub
}

void ScopedSpan::SetAttribute([[maybe_unused]] const std::string& key,
                              [[maybe_unused]] bool value) {
  // Stub
}

void ScopedSpan::AddEvent([[maybe_unused]] const std::string& name) {
  // Stub
}

void ScopedSpan::RecordError([[maybe_unused]] const std::string& error_message) {
  // Stub
}

}  // namespace jrb::core::telemetry

#include <spdlog/spdlog.h>

#include "cpp_accelerator/core/telemetry.h"

// Stub implementation for when OpenTelemetry C++ dependencies are not available
// This allows the code to compile while we work on resolving OpenTelemetry C++ dependencies

namespace jrb::core::telemetry {

TelemetryManager& TelemetryManager::GetInstance() {
    static TelemetryManager instance;
    return instance;
}

bool TelemetryManager::Initialize(const std::string& service_name,
                                  const std::string& collector_endpoint, bool enabled) {
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

TracerPtr TelemetryManager::GetTracer(const std::string& name) {
    return nullptr;
}

SpanPtr TelemetryManager::CreateSpan(const std::string& tracer_name, const std::string& span_name) {
    return std::shared_ptr<StubSpan>(new StubSpan());
}

SpanPtr TelemetryManager::CreateChildSpan(const std::string& tracer_name,
                                          const std::string& span_name,
                                          const std::string& parent_trace_id,
                                          const std::string& parent_span_id, uint8_t trace_flags) {
    return std::shared_ptr<StubSpan>(new StubSpan());
}

void ScopedSpan::SetAttribute(const std::string& key, const std::string& value) {
    // Stub
}

void ScopedSpan::SetAttribute(const std::string& key, int64_t value) {
    // Stub
}

void ScopedSpan::SetAttribute(const std::string& key, double value) {
    // Stub
}

void ScopedSpan::SetAttribute(const std::string& key, bool value) {
    // Stub
}

void ScopedSpan::AddEvent(const std::string& name) {
    // Stub
}

void ScopedSpan::RecordError(const std::string& error_message) {
    // Stub
}

}  // namespace jrb::core::telemetry

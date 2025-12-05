#pragma once

#include <memory>
#include <string>

namespace jrb::core::telemetry {

// Stub types when OpenTelemetry C++ is not available
struct StubTracer {};
struct StubSpan {
  void End() {}
};

using TracerPtr = std::shared_ptr<StubTracer>;
using SpanPtr = std::shared_ptr<StubSpan>;

class TelemetryManager {
public:
  static TelemetryManager& GetInstance();

  bool Initialize(const std::string& service_name, const std::string& collector_endpoint,
                  bool enabled = true);

  void Shutdown();

  TracerPtr GetTracer(const std::string& name);

  bool IsEnabled() const { return enabled_; }

  SpanPtr CreateSpan(const std::string& tracer_name, const std::string& span_name);

  SpanPtr CreateChildSpan(const std::string& tracer_name, const std::string& span_name,
                          const std::string& parent_trace_id, const std::string& parent_span_id,
                          uint8_t trace_flags);

private:
  TelemetryManager() = default;
  ~TelemetryManager() = default;

  TelemetryManager(const TelemetryManager&) = delete;
  TelemetryManager& operator=(const TelemetryManager&) = delete;

  bool enabled_ = false;
  bool initialized_ = false;
};

class ScopedSpan {
public:
  explicit ScopedSpan(SpanPtr span) : span_(span) {}

  ~ScopedSpan() {
    if (span_) {
      span_->End();
    }
  }

  SpanPtr Get() const { return span_; }

  void SetAttribute(const std::string& key, const std::string& value);
  void SetAttribute(const std::string& key, int64_t value);
  void SetAttribute(const std::string& key, double value);
  void SetAttribute(const std::string& key, bool value);

  void AddEvent(const std::string& name);
  void RecordError(const std::string& error_message);

private:
  SpanPtr span_;
};

}  // namespace jrb::core::telemetry

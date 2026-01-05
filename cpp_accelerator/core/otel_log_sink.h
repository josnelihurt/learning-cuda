#pragma once

#include <memory>
#include <mutex>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/details/log_msg.h>
#include <spdlog/sinks/base_sink.h>
#pragma GCC diagnostic pop

namespace jrb::core {

// Forward declarations
class OtelLogSinkImpl;

// Custom spdlog sink that forwards logs to OpenTelemetry
class OtelLogSink : public spdlog::sinks::base_sink<std::mutex> {
public:
  OtelLogSink(const std::string& endpoint, const std::string& environment);
  ~OtelLogSink() override;

  // Disable copy and move
  OtelLogSink(const OtelLogSink&) = delete;
  OtelLogSink& operator=(const OtelLogSink&) = delete;
  OtelLogSink(OtelLogSink&&) = delete;
  OtelLogSink& operator=(OtelLogSink&&) = delete;

protected:
  void sink_it_(const spdlog::details::log_msg& msg) override;
  void flush_() override;

private:
  std::unique_ptr<OtelLogSinkImpl> pimpl_;
};

}  // namespace jrb::core

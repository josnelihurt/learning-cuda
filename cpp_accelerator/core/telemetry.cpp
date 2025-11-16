#include "cpp_accelerator/core/telemetry.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "opentelemetry/exporters/otlp/otlp_grpc_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_grpc_exporter_options.h"
#include "opentelemetry/sdk/resource/resource.h"
#include "opentelemetry/sdk/resource/semantic_conventions.h"
#include "opentelemetry/sdk/trace/batch_span_processor_factory.h"
#include "opentelemetry/sdk/trace/batch_span_processor_options.h"
#include "opentelemetry/sdk/trace/processor.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"

namespace jrb::core::telemetry {

namespace trace_sdk = opentelemetry::sdk::trace;
namespace trace_api = opentelemetry::trace;
namespace otlp = opentelemetry::exporter::otlp;
namespace resource = opentelemetry::sdk::resource;

TelemetryManager& TelemetryManager::GetInstance() {
    static TelemetryManager instance;
    return instance;
}

bool TelemetryManager::Initialize(const std::string& service_name,
                                  const std::string& collector_endpoint, bool enabled) {
    if (initialized_) {
        spdlog::warn("TelemetryManager already initialized");
        return true;
    }

    enabled_ = enabled;
    if (!enabled_) {
        spdlog::info("Telemetry disabled by configuration");
        initialized_ = true;
        return true;
    }

    try {
        otlp::OtlpGrpcExporterOptions opts;
        opts.endpoint = collector_endpoint;
        opts.use_ssl_credentials = false;

        auto exporter = otlp::OtlpGrpcExporterFactory::Create(opts);

        trace_sdk::BatchSpanProcessorOptions processor_opts;
        auto processor =
            trace_sdk::BatchSpanProcessorFactory::Create(std::move(exporter), processor_opts);

        auto resource_attributes = resource::ResourceAttributes{
            {resource::SemanticConventions::kServiceName, service_name},
            {resource::SemanticConventions::kServiceVersion, "1.0.0"}};
        auto resource_ptr = resource::Resource::Create(resource_attributes);

        auto provider =
            trace_sdk::TracerProviderFactory::Create(std::move(processor), resource_ptr);

        trace_api::Provider::SetTracerProvider(provider);

        spdlog::info("OpenTelemetry C++ SDK initialized (endpoint: {}, service: {})",
                     collector_endpoint, service_name);
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize OpenTelemetry: {}", e.what());
        enabled_ = false;
        return false;
    }
}

void TelemetryManager::Shutdown() {
    if (!initialized_) {
        return;
    }

    auto provider = trace_api::Provider::GetTracerProvider();
    if (auto sdk_provider = std::dynamic_pointer_cast<trace_sdk::TracerProvider>(provider)) {
        sdk_provider->Shutdown();
        spdlog::info("OpenTelemetry C++ SDK shut down");
    }
}

TracerPtr TelemetryManager::GetTracer(const std::string& name) {
    if (!enabled_) {
        return trace_api::Provider::GetTracerProvider()->GetTracer(name);
    }
    return trace_api::Provider::GetTracerProvider()->GetTracer(name);
}

SpanPtr TelemetryManager::CreateSpan(const std::string& tracer_name, const std::string& span_name) {
    auto tracer = GetTracer(tracer_name);
    trace_api::StartSpanOptions opts;
    opts.kind = trace_api::SpanKind::kInternal;
    return tracer->StartSpan(span_name, opts);
}

SpanPtr TelemetryManager::CreateChildSpan(const std::string& tracer_name,
                                          const std::string& span_name,
                                          const std::string& parent_trace_id,
                                          const std::string& parent_span_id, uint8_t trace_flags) {
    auto tracer = GetTracer(tracer_name);

    if (parent_trace_id.empty() || parent_span_id.empty()) {
        return CreateSpan(tracer_name, span_name);
    }

    try {
        trace_api::TraceId trace_id;
        trace_api::SpanId span_id;

        trace_id.CopyFromHex(parent_trace_id);
        span_id.CopyFromHex(parent_span_id);

        trace_api::SpanContext parent_context(trace_id, span_id, trace_api::TraceFlags(trace_flags),
                                              true);

        trace_api::StartSpanOptions opts;
        opts.kind = trace_api::SpanKind::kInternal;
        opts.parent = parent_context;

        return tracer->StartSpan(span_name, opts);
    } catch (const std::exception& e) {
        spdlog::warn("Failed to create child span with parent context: {}", e.what());
        return CreateSpan(tracer_name, span_name);
    }
}

void ScopedSpan::SetAttribute(const std::string& key, const std::string& value) {
    if (span_) {
        span_->SetAttribute(key, value);
    }
}

void ScopedSpan::SetAttribute(const std::string& key, int64_t value) {
    if (span_) {
        span_->SetAttribute(key, value);
    }
}

void ScopedSpan::SetAttribute(const std::string& key, double value) {
    if (span_) {
        span_->SetAttribute(key, value);
    }
}

void ScopedSpan::SetAttribute(const std::string& key, bool value) {
    if (span_) {
        span_->SetAttribute(key, value);
    }
}

void ScopedSpan::AddEvent(const std::string& name) {
    if (span_) {
        span_->AddEvent(name);
    }
}

void ScopedSpan::RecordError(const std::string& error_message) {
    if (span_) {
        span_->SetStatus(trace_api::StatusCode::kError, error_message);
        span_->AddEvent("exception", {{"exception.message", error_message}});
    }
}

}  // namespace jrb::core::telemetry

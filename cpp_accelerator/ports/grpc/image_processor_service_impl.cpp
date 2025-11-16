#include "cpp_accelerator/ports/grpc/image_processor_service_impl.h"

#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <grpcpp/server_context.h>
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::ports::grpc_service {

namespace {

cuda_learning::GenericFilterParameterType ConvertParameterType(const std::string& type) {
  if (type == "select") {
    return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_SELECT;
  }
  if (type == "range") {
    return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_RANGE;
  }
  if (type == "number") {
    return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_NUMBER;
  }
  if (type == "checkbox") {
    return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_CHECKBOX;
  }
  if (type == "text") {
    return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_TEXT;
  }
  return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_UNSPECIFIED;
}

::grpc::Status EngineErrorStatus(const cuda_learning::ProcessImageResponse& response) {
  auto code = response.code();
  if (code == 0) {
    return ::grpc::Status::OK;
  }
  auto status_code = ::grpc::StatusCode::INTERNAL;
  if (code == 1) {
    status_code = ::grpc::StatusCode::INVALID_ARGUMENT;
  } else if (code == 5) {
    status_code = ::grpc::StatusCode::FAILED_PRECONDITION;
  } else if (code == 6) {
    status_code = ::grpc::StatusCode::INTERNAL;
  } else if (code == 7) {
    status_code = ::grpc::StatusCode::FAILED_PRECONDITION;
  }
  return ::grpc::Status(status_code, response.message());
}

}  // namespace

ImageProcessorServiceImpl::ImageProcessorServiceImpl(std::shared_ptr<ProcessorEngineProvider> engine)
    : engine_(std::move(engine)) {}

::grpc::Status ImageProcessorServiceImpl::ProcessImage(
    ::grpc::ServerContext* /*context*/, const cuda_learning::ProcessImageRequest* request,
    cuda_learning::ProcessImageResponse* response) {
  if (!EnsureEngine() || request == nullptr || response == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION, "Processor engine not available");
  }

  CopyProcessMetadata(*request, response);

  bool ok = engine_->ProcessImage(*request, response);
  if (!ok || response->code() != 0) {
    spdlog::warn("ProcessImage failed via gRPC: {}", response->message());
    return EngineFailureStatus(*response);
  }

  return ::grpc::Status::OK;
}

::grpc::Status ImageProcessorServiceImpl::ListFilters(::grpc::ServerContext* /*context*/,
                                                      const cuda_learning::ListFiltersRequest* request,
                                                      cuda_learning::ListFiltersResponse* response) {
  if (!EnsureEngine() || request == nullptr || response == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION, "Processor engine not available");
  }

  response->clear_filters();
  CopyTraceContext(request->trace_context(), response->mutable_trace_context());
  response->set_api_version(request->api_version());

  PopulateListFiltersResponse(response);
  return ::grpc::Status::OK;
}

::grpc::Status ImageProcessorServiceImpl::StreamProcessVideo(
    ::grpc::ServerContext* /*context*/,
    ::grpc::ServerReaderWriter<cuda_learning::ProcessImageResponse,
                               cuda_learning::ProcessImageRequest>* stream) {
  if (!EnsureEngine() || stream == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION, "Processor engine not available");
  }

  cuda_learning::ProcessImageRequest request;
  while (stream->Read(&request)) {
    cuda_learning::ProcessImageResponse response;
    CopyProcessMetadata(request, &response);

  bool ok = engine_->ProcessImage(request, &response);
    if (!ok || response.code() != 0) {
      spdlog::warn("StreamProcessVideo frame failed (code={}): {}", response.code(),
                   response.message());
      stream->Write(response);
      return EngineFailureStatus(response);
    }

    stream->Write(response);
  }

  return ::grpc::Status::OK;
}

bool ImageProcessorServiceImpl::EnsureEngine() const { return static_cast<bool>(engine_); }

void ImageProcessorServiceImpl::CopyTraceContext(const cuda_learning::TraceContext& source,
                                                 cuda_learning::TraceContext* target) const {
  if (!target) {
    return;
  }
  target->CopyFrom(source);
}

void ImageProcessorServiceImpl::CopyProcessMetadata(
    const cuda_learning::ProcessImageRequest& request,
    cuda_learning::ProcessImageResponse* response) const {
  if (!response) {
    return;
  }
  response->set_api_version(request.api_version());
  CopyTraceContext(request.trace_context(), response->mutable_trace_context());
}

::grpc::Status ImageProcessorServiceImpl::EngineFailureStatus(
    const cuda_learning::ProcessImageResponse& response) const {
  return EngineErrorStatus(response);
}

void ImageProcessorServiceImpl::PopulateListFiltersResponse(
    cuda_learning::ListFiltersResponse* response) const {
  if (!response || !EnsureEngine()) {
    return;
  }

  cuda_learning::GetCapabilitiesResponse caps_response;
  engine_->GetCapabilities(&caps_response);
  const auto& caps = caps_response.capabilities();

  response->set_api_version(caps.api_version());
  for (const auto& filter : caps.filters()) {
    auto* generic_filter = response->add_filters();
    generic_filter->set_id(filter.id());
    generic_filter->set_name(filter.name());

    for (const auto& param : filter.parameters()) {
      auto* generic_param = generic_filter->add_parameters();
      generic_param->set_id(param.id());
      generic_param->set_name(param.name());
      generic_param->set_type(ConvertParameterType(param.type()));
      generic_param->set_default_value(param.default_value());
      for (const auto& option : param.options()) {
        auto* generic_option = generic_param->add_options();
        generic_option->set_value(option);
        generic_option->set_label(option);
      }
    }

    for (const auto accelerator : filter.supported_accelerators()) {
      generic_filter->add_supported_accelerators(
          static_cast<cuda_learning::AcceleratorType>(accelerator));
    }
  }
}

}  // namespace jrb::ports::grpc_service



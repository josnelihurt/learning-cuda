#include "cpp_accelerator/ports/cgo/cgo_api.h"

#include <cstring>
#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#pragma GCC diagnostic pop

#include "cpp_accelerator/core/telemetry.h"
#include "cpp_accelerator/ports/cgo/image_buffer_adapter.h"
#include "cpp_accelerator/ports/shared_lib/processor_engine.h"

// TODO(deprecate-cgo): Remove this CGO transport layer once the gRPC service fully replaces it.

namespace {

uint8_t* allocate_response(const std::string& serialized, int* out_len) {
  *out_len = static_cast<int>(serialized.size());
  uint8_t* buffer = new uint8_t[*out_len];
  std::memcpy(buffer, serialized.data(), *out_len);
  return buffer;
}

jrb::infrastructure::cpu::BorderMode proto_to_border_mode(cuda_learning::BorderMode mode) {
  switch (mode) {
    case cuda_learning::BORDER_MODE_CLAMP:
      return jrb::infrastructure::cpu::BorderMode::CLAMP;
    case cuda_learning::BORDER_MODE_REFLECT:
      return jrb::infrastructure::cpu::BorderMode::REFLECT;
    case cuda_learning::BORDER_MODE_WRAP:
      return jrb::infrastructure::cpu::BorderMode::WRAP;
    default:
      return jrb::infrastructure::cpu::BorderMode::REFLECT;
  }
}

jrb::infrastructure::cuda::BorderMode proto_to_cuda_border_mode(cuda_learning::BorderMode mode) {
  switch (mode) {
    case cuda_learning::BORDER_MODE_CLAMP:
      return jrb::infrastructure::cuda::BorderMode::CLAMP;
    case cuda_learning::BORDER_MODE_REFLECT:
      return jrb::infrastructure::cuda::BorderMode::REFLECT;
    case cuda_learning::BORDER_MODE_WRAP:
      return jrb::infrastructure::cuda::BorderMode::WRAP;
    default:
      return jrb::infrastructure::cuda::BorderMode::REFLECT;
  }
}

}  // anonymous namespace

extern "C" {

bool CudaInit(const uint8_t* request, int request_len, uint8_t** response, int* response_len) {
  static jrb::ports::shared_lib::ProcessorEngine engine("cgo-api");
  cuda_learning::InitRequest init_req;
  cuda_learning::InitResponse init_resp;

  // Parse request
  if (!init_req.ParseFromArray(request, request_len)) {
    spdlog::error("Failed to parse InitRequest");
    init_resp.set_code(1);
    init_resp.set_message("Failed to parse InitRequest");
    *response = allocate_response(init_resp.SerializeAsString(), response_len);
    return false;
  }

  bool ok = engine.Initialize(init_req, &init_resp);
  *response = allocate_response(init_resp.SerializeAsString(), response_len);
  return ok;
}

void CudaCleanup() {
  spdlog::info("Cleaning up CUDA context and telemetry");

  auto& telemetry = jrb::core::telemetry::TelemetryManager::GetInstance();
  telemetry.Shutdown();
}

bool ProcessImage(const uint8_t* request, int request_len, uint8_t** response, int* response_len) {
  static jrb::ports::shared_lib::ProcessorEngine engine("cgo-api");
  cuda_learning::ProcessImageRequest proc_req;
  cuda_learning::ProcessImageResponse proc_resp;

  // Parse request
  if (!proc_req.ParseFromArray(request, request_len)) {
    spdlog::error("Failed to parse ProcessImageRequest");
    proc_resp.set_code(1);
    proc_resp.set_message("Failed to parse ProcessImageRequest");
    *response = allocate_response(proc_resp.SerializeAsString(), response_len);
    return false;
  }

  bool ok = engine.ProcessImage(proc_req, &proc_resp);
  *response = allocate_response(proc_resp.SerializeAsString(), response_len);
  return ok;
}

bool GetCapabilities(const uint8_t* request, int request_len, uint8_t** response,
                     int* response_len) {
  static jrb::ports::shared_lib::ProcessorEngine engine("cgo-api");
  (void)request;
  (void)request_len;

  cuda_learning::GetCapabilitiesResponse resp;
  engine.GetCapabilities(&resp);
  *response = allocate_response(resp.SerializeAsString(), response_len);
  return true;
}

void FreeResponse(uint8_t* ptr) {
  delete[] ptr;
}

}  // extern "C"

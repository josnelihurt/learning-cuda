#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#pragma GCC diagnostic pop

#include "cpp_accelerator/core/logger.h"
#include "cpp_accelerator/core/telemetry.h"
#include "cpp_accelerator/ports/shared_lib/library_version.h"
#include "cpp_accelerator/ports/shared_lib/processor_api.h"
#include "cpp_accelerator/ports/shared_lib/processor_engine.h"
#include "image_buffer_adapter.h"

namespace {

// Helper to allocate response buffer (must be freed by caller with FreeResponse)
uint8_t* allocate_response(const std::string& serialized, int* out_len) {
  *out_len = static_cast<int>(serialized.size());
  uint8_t* buffer = new uint8_t[*out_len];
  std::memcpy(buffer, serialized.data(), *out_len);
  return buffer;
}

}  // anonymous namespace

extern "C" {

processor_version_t processor_api_version(void) {
  processor_version_t version;
  version.major = (PROCESSOR_API_VERNUM >> 16) & 0xFF;
  version.minor = (PROCESSOR_API_VERNUM >> 8) & 0xFF;
  version.patch = PROCESSOR_API_VERNUM & 0xFF;
  return version;
}

bool processor_init(const uint8_t* request, int request_len, uint8_t** response,
                    int* response_len) {
  static jrb::ports::shared_lib::ProcessorEngine engine("cgo-api");
  static bool logger_initialized = false;
  if (!logger_initialized) {
    // Initialize logger BEFORE any other spdlog calls
    jrb::core::initialize_logger();

    // Verify logger is active and flush to ensure file is created
    auto logger = spdlog::default_logger();
    if (logger) {
      logger->flush();
    }

    logger_initialized = true;
  }

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

  spdlog::info("Initializing CUDA context (device: {})", init_req.cuda_device_id());

  bool ok = engine.Initialize(init_req, &init_resp);
  *response = allocate_response(init_resp.SerializeAsString(), response_len);

  return ok;
}

void processor_cleanup() {
  spdlog::info("Cleaning up CUDA context and telemetry");

  auto& telemetry = jrb::core::telemetry::TelemetryManager::GetInstance();
  telemetry.Shutdown();
}

bool processor_process_image(const uint8_t* request, int request_len, uint8_t** response,
                             int* response_len) {
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

bool processor_get_capabilities(const uint8_t* request, int request_len, uint8_t** response,
                                int* response_len) {
  static jrb::ports::shared_lib::ProcessorEngine engine("cgo-api");
  (void)request;
  (void)request_len;

  cuda_learning::GetCapabilitiesResponse resp;
  engine.GetCapabilities(&resp);
  auto* caps = resp.mutable_capabilities();
  char version_buf[64];
  if (processor_get_library_version(version_buf, sizeof(version_buf))) {
    caps->set_library_version(version_buf);
  }
  *response = allocate_response(resp.SerializeAsString(), response_len);
  return true;
}

bool processor_get_library_version(char* version_buf, int buf_len) {
  spdlog::info("GetLibraryVersion request received - fetching library version");

  if (!version_buf || buf_len < 1) {
    spdlog::error("Invalid buffer for GetLibraryVersion");
    return false;
  }

  version_buf[0] = '\0';

  std::string version = LIBRARY_VERSION_STR;
  spdlog::info("Library version from embedded VERSION file: {}", version);

  size_t copy_len = version.size();
  if (copy_len >= static_cast<size_t>(buf_len)) {
    copy_len = static_cast<size_t>(buf_len) - 1;
  }

  std::strncpy(version_buf, version.c_str(), copy_len);
  version_buf[copy_len] = '\0';

  spdlog::info("GetLibraryVersion completed - returning version: {}", version_buf);
  return true;
}

void processor_free_response(uint8_t* ptr) {
  delete[] ptr;
}

}  // extern "C"

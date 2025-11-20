#include "cpp_accelerator/ports/grpc/processor_engine_adapter.h"

#include <fstream>
#include <string>

namespace jrb::ports::grpc_service {

ProcessorEngineAdapter::ProcessorEngineAdapter(
    std::shared_ptr<jrb::ports::shared_lib::ProcessorEngine> engine)
    : engine_(std::move(engine)) {}

bool ProcessorEngineAdapter::ProcessImage(const cuda_learning::ProcessImageRequest& request,
                                          cuda_learning::ProcessImageResponse* response) {
  if (!engine_ || !response) {
    return false;
  }
  return engine_->ProcessImage(request, response);
}

bool ProcessorEngineAdapter::GetCapabilities(cuda_learning::GetCapabilitiesResponse* response) {
  if (!engine_ || !response) {
    return false;
  }
  return engine_->GetCapabilities(response);
}

bool ProcessorEngineAdapter::GetVersionInfo(cuda_learning::GetVersionInfoResponse* response) {
  if (!engine_ || !response) {
    return false;
  }

  std::string server_version;
  static const char* version_file_paths[] = {"cpp_accelerator/VERSION", "../cpp_accelerator/VERSION",
                                              "../../cpp_accelerator/VERSION", "./VERSION", nullptr};

  bool found = false;
  for (int i = 0; version_file_paths[i] != nullptr && !found; ++i) {
    std::ifstream file(version_file_paths[i]);
    if (file.is_open()) {
      std::getline(file, server_version);
      file.close();
      if (!server_version.empty()) {
        found = true;
      }
    }
  }

  if (!found) {
    server_version = "unknown";
  }

  response->set_server_version(server_version);

  cuda_learning::GetCapabilitiesResponse caps_response;
  if (engine_->GetCapabilities(&caps_response)) {
    const auto& caps = caps_response.capabilities();
    response->set_library_version(caps.library_version());
    response->set_build_date(caps.build_date());
    response->set_build_commit(caps.build_commit());
  } else {
    response->set_library_version("unknown");
    response->set_build_date("unknown");
    response->set_build_commit("unknown");
  }

  response->set_code(0);
  response->set_message("OK");
  return true;
}

}  // namespace jrb::ports::grpc_service



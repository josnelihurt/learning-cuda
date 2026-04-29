#include "src/cpp_accelerator/application/server_info/server_info_provider.h"

#include <fstream>
#include <string>

#include "src/cpp_accelerator/application/engine/processor_engine.h"
#include "src/cpp_accelerator/application/server_info/filter_parameter_mapping.h"

namespace jrb::application::server_info {

ServerInfoProvider::ServerInfoProvider(jrb::application::engine::ProcessorEngine* engine)
    : engine_(engine) {}

void ServerInfoProvider::PopulateVersionResponse(cuda_learning::GetVersionInfoResponse* response) {
  if (response == nullptr) return;
  if (engine_ == nullptr) {
    response->set_code(6);
    response->set_message("engine unavailable");
    return;
  }

  std::string server_version;
  static const char* kVersionFilePaths[] = {"src/cpp_accelerator/VERSION",
                                            "../cpp_accelerator/VERSION",
                                            "../../cpp_accelerator/VERSION", "./VERSION", nullptr};
  for (int i = 0; kVersionFilePaths[i] != nullptr; ++i) {
    std::ifstream file(kVersionFilePaths[i]);
    if (file.is_open()) {
      std::getline(file, server_version);
      if (!server_version.empty()) break;
    }
  }
  if (server_version.empty()) server_version = "unknown";
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
}

void ServerInfoProvider::PopulateListFiltersResponse(
    const cuda_learning::ListFiltersRequest& request,
    cuda_learning::ListFiltersResponse* response) {
  response->set_api_version(request.api_version());
  if (engine_ == nullptr) return;

  cuda_learning::GetCapabilitiesResponse caps;
  if (!engine_->GetCapabilities(&caps, request.requested_accelerator())) return;

  for (const auto& filter : caps.capabilities().filters()) {
    auto* gf = response->add_filters();
    gf->set_id(filter.id());
    gf->set_name(filter.name());
    for (const auto& param : filter.parameters()) {
      auto* gp = gf->add_parameters();
      gp->set_id(param.id());
      gp->set_name(param.name());
      gp->set_type(ConvertParamType(param.type()));
      gp->set_default_value(param.default_value());
      CopyValidationRulesFromMetadata(param, gp);
      for (const auto& opt : param.options()) {
        auto* go = gp->add_options();
        go->set_value(opt);
        go->set_label(opt);
      }
    }
    for (const auto acc : filter.supported_accelerators()) {
      gf->add_supported_accelerators(static_cast<cuda_learning::AcceleratorType>(acc));
    }
  }
}

}  // namespace jrb::application::server_info

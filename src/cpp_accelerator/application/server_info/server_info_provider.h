#pragma once

#include "src/cpp_accelerator/application/server_info/i_server_info_provider.h"

namespace jrb::application::engine {
class ProcessorEngine;
}

namespace jrb::application::server_info {

class ServerInfoProvider : public IServerInfoProvider {
 public:
  explicit ServerInfoProvider(jrb::application::engine::ProcessorEngine* engine);

  void PopulateVersionResponse(cuda_learning::GetVersionInfoResponse* response) override;

  void PopulateListFiltersResponse(const cuda_learning::ListFiltersRequest& request,
                                   cuda_learning::ListFiltersResponse* response) override;

 private:
  jrb::application::engine::ProcessorEngine* engine_;
};

}  // namespace jrb::application::server_info

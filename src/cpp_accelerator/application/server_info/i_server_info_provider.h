#pragma once

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"

namespace jrb::application::server_info {

class IServerInfoProvider {
 public:
  virtual ~IServerInfoProvider() = default;

  virtual void PopulateVersionResponse(cuda_learning::GetVersionInfoResponse* response) = 0;

  virtual void PopulateListFiltersResponse(const cuda_learning::ListFiltersRequest& request,
                                           cuda_learning::ListFiltersResponse* response) = 0;
};

}  // namespace jrb::application::server_info

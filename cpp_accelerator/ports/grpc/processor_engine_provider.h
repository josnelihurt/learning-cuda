#pragma once

#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#pragma GCC diagnostic pop

namespace jrb::ports::grpc_service {

class ProcessorEngineProvider {
 public:
  virtual ~ProcessorEngineProvider() = default;

  virtual bool ProcessImage(const cuda_learning::ProcessImageRequest& request,
                            cuda_learning::ProcessImageResponse* response) = 0;

  virtual bool GetCapabilities(cuda_learning::GetCapabilitiesResponse* response) = 0;
};

}  // namespace jrb::ports::grpc_service



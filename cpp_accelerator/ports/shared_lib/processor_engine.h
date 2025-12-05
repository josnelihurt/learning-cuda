#pragma once

#include <memory>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#pragma GCC diagnostic pop

#include "cpp_accelerator/domain/interfaces/grayscale_algorithm.h"
#include "processor_api.h"

namespace jrb::ports::shared_lib {

class ProcessorEngine {
public:
  explicit ProcessorEngine(std::string component_name = "processor-engine");
  ~ProcessorEngine() = default;

  bool Initialize(const cuda_learning::InitRequest& request, cuda_learning::InitResponse* response);

  bool ProcessImage(const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response);

  bool GetCapabilities(cuda_learning::GetCapabilitiesResponse* response);

private:
  jrb::domain::interfaces::GrayscaleAlgorithm ProtoToAlgorithm(
      cuda_learning::GrayscaleType type) const;
  bool ApplyFilters(const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response);

  std::string component_name_;
};

}  // namespace jrb::ports::shared_lib

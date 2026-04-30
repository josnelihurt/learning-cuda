#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/application/engine/filter_factory_registry.h"
#include "src/cpp_accelerator/domain/interfaces/grayscale_algorithm.h"

namespace jrb::application::engine {

class ProcessorEngine {
public:
  explicit ProcessorEngine(std::string component_name = "processor-engine");
  ~ProcessorEngine() = default;

  bool Initialize(const cuda_learning::InitRequest& request, cuda_learning::InitResponse* response);

  bool ProcessImage(const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response,
                    void* memory_pool = nullptr);

  bool GetCapabilities(cuda_learning::GetCapabilitiesResponse* response,
                       cuda_learning::AcceleratorType requested_accelerator =
                           cuda_learning::ACCELERATOR_TYPE_UNSPECIFIED);

private:
  jrb::domain::interfaces::GrayscaleAlgorithm ProtoToAlgorithm(
      cuda_learning::GrayscaleType type) const;
  bool ApplyFilters(const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response,
                    void* memory_pool = nullptr);

  std::string component_name_;
  FilterFactoryRegistry factory_registry_;
};

}  // namespace jrb::application::engine

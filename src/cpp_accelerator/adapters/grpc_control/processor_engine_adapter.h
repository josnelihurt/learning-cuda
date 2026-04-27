#pragma once

#include <memory>

#include "src/cpp_accelerator/application/engine/processor_engine.h"
#include "src/cpp_accelerator/adapters/grpc_control/processor_engine_provider.h"

namespace jrb::adapters::grpc_control {

class ProcessorEngineAdapter : public ProcessorEngineProvider {
public:
  explicit ProcessorEngineAdapter(std::shared_ptr<jrb::application::engine::ProcessorEngine> engine);
  ~ProcessorEngineAdapter() override = default;

  bool ProcessImage(const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response) override;

  bool GetCapabilities(cuda_learning::GetCapabilitiesResponse* response) override;

  bool GetVersionInfo(cuda_learning::GetVersionInfoResponse* response) override;

  std::shared_ptr<jrb::application::engine::ProcessorEngine> underlying() const { return engine_; }

private:
  std::shared_ptr<jrb::application::engine::ProcessorEngine> engine_;
};

}  // namespace jrb::adapters::grpc_control

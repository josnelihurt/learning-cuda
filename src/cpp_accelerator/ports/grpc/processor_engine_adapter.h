#pragma once

#include <memory>

#include "cpp_accelerator/ports/grpc/processor_engine_provider.h"
#include "cpp_accelerator/ports/shared_lib/processor_engine.h"

namespace jrb::ports::grpc_service {

class ProcessorEngineAdapter : public ProcessorEngineProvider {
public:
  explicit ProcessorEngineAdapter(std::shared_ptr<jrb::ports::shared_lib::ProcessorEngine> engine);
  ~ProcessorEngineAdapter() override = default;

  bool ProcessImage(const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response) override;

  bool GetCapabilities(cuda_learning::GetCapabilitiesResponse* response) override;

  bool GetVersionInfo(cuda_learning::GetVersionInfoResponse* response) override;

  std::shared_ptr<jrb::ports::shared_lib::ProcessorEngine> underlying() const { return engine_; }

private:
  std::shared_ptr<jrb::ports::shared_lib::ProcessorEngine> engine_;
};

}  // namespace jrb::ports::grpc_service

#include "cpp_accelerator/ports/grpc/processor_engine_adapter.h"

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

}  // namespace jrb::ports::grpc_service



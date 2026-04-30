#include "src/cpp_accelerator/composition/platform/platform_support.h"

#include "src/cpp_accelerator/composition/platform/cpu/cpu_platform.h"
#include "src/cpp_accelerator/composition/platform/cuda/cuda_platform.h"

namespace jrb::application::engine {

void RegisterPlatformAccelerators(FilterFactoryRegistry& registry) {
  platform::cpu::RegisterFactories(registry);
  platform::cuda::RegisterFactories(registry);
}

void InitializePlatformSubsystems(const cuda_learning::InitRequest& request,
                                  cuda_learning::InitResponse* response) {
  platform::cuda::Initialize(request, response);
}

bool ApplyInference(const std::string& model_id, float confidence, bool pipeline_has_output,
                    const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response,
                    const jrb::domain::interfaces::ImageBuffer& input_buffer,
                    jrb::domain::interfaces::ImageBufferMut& output_buffer) {
  return platform::cuda::ApplyInference(model_id, confidence, pipeline_has_output, request,
                                        response, input_buffer, output_buffer);
}

}  // namespace jrb::application::engine

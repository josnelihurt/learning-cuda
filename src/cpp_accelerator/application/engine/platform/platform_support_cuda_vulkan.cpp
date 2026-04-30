#include "src/cpp_accelerator/application/engine/platform/platform_support.h"

#include "src/cpp_accelerator/adapters/compute/cpu/cpu_filter_factory.h"
#include "src/cpp_accelerator/application/engine/platform/cuda/cuda_platform.h"
#include "src/cpp_accelerator/application/engine/platform/vulkan/vulkan_platform.h"

namespace jrb::application::engine {

void RegisterPlatformAccelerators(FilterFactoryRegistry& registry) {
  registry.Register(std::make_unique<jrb::infrastructure::cpu::CpuFilterFactory>());
  platform::cuda::RegisterFactories(registry);
  platform::vulkan::RegisterFactories(registry);
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

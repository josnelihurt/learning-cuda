#pragma once

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/application/engine/filter_factory_registry.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::application::engine {

// Registers all platform-specific filter factories into the registry.
// CPU factory is always registered; additional factories depend on the build profile.
void RegisterPlatformAccelerators(FilterFactoryRegistry& registry);

// Performs platform-specific initialization (e.g., TRT engine preload on CUDA builds).
// On CPU-only builds this is a no-op.
void InitializePlatformSubsystems(const cuda_learning::InitRequest& request,
                                  cuda_learning::InitResponse* response);

// Applies model inference if a matching detector is available.
// If pipeline_has_output is false and inference is applied, copies input_buffer to output_buffer
// so the original frame is preserved as output while detections are annotated.
// Returns true if inference was applied, false if no detector was available.
bool ApplyInference(const std::string& model_id, float confidence, bool pipeline_has_output,
                    const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response,
                    const jrb::domain::interfaces::ImageBuffer& input_buffer,
                    jrb::domain::interfaces::ImageBufferMut& output_buffer);

}  // namespace jrb::application::engine

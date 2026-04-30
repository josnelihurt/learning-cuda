#pragma once

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/application/engine/filter_factory_registry.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::application::engine::platform::cuda {

void RegisterFactories(FilterFactoryRegistry& registry);

void Initialize(const cuda_learning::InitRequest& request, cuda_learning::InitResponse* response);

bool ApplyInference(const std::string& model_id, float confidence, bool pipeline_has_output,
                    const cuda_learning::ProcessImageRequest& request,
                    cuda_learning::ProcessImageResponse* response,
                    const jrb::domain::interfaces::ImageBuffer& input_buffer,
                    jrb::domain::interfaces::ImageBufferMut& output_buffer);

}  // namespace jrb::application::engine::platform::cuda

#pragma once

#include <optional>
#include <string>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"

namespace jrb::adapters::webrtc::protocol {

std::string NormalizeFilterId(const std::string& value);

std::optional<std::string> FirstGenericValue(
    const cuda_learning::GenericFilterParameterSelection& selection);

cuda_learning::GrayscaleType MapStringToGrayscaleType(const std::string& value);
cuda_learning::BorderMode MapStringToBorderMode(const std::string& value);
bool ParseBool(const std::string& value);

void ResolveGenericSelectionsInPlace(cuda_learning::ProcessImageRequest* request);

}  // namespace jrb::adapters::webrtc::protocol

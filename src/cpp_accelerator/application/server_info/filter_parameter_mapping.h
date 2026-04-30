#pragma once

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"

namespace jrb::application::server_info {

// Engine FilterParameter (rich metadata) -> wire GenericFilterParameterType.
cuda_learning::GenericFilterParameterType ConvertParamType(const std::string& type);

// Lifts validation rules from FilterParameter::metadata (typed values like
// min/max/step/required/min_items/max_items/pattern) into typed fields of
// GenericFilterParameter, while keeping the raw metadata map intact.
void CopyValidationRulesFromMetadata(const cuda_learning::FilterParameter& source,
                                     cuda_learning::GenericFilterParameter* target);

}  // namespace jrb::application::server_info

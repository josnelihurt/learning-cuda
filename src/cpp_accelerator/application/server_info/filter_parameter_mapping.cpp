#include "src/cpp_accelerator/application/server_info/filter_parameter_mapping.h"

#include <exception>
#include <string>

namespace jrb::application::server_info {

cuda_learning::GenericFilterParameterType ConvertParamType(const std::string& type) {
  if (type == "select")   return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_SELECT;
  if (type == "range")    return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_RANGE;
  if (type == "number")   return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_NUMBER;
  if (type == "checkbox") return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_CHECKBOX;
  if (type == "text")     return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_TEXT;
  return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_UNSPECIFIED;
}

void CopyValidationRulesFromMetadata(const cuda_learning::FilterParameter& source,
                                     cuda_learning::GenericFilterParameter* target) {
  if (target == nullptr) return;
  for (const auto& [key, value] : source.metadata()) {
    (*target->mutable_metadata())[key] = value;
    try {
      if (key == "required") {
        target->set_required(value == "true" || value == "1");
      } else if (key == "min") {
        target->set_min_value(std::stod(value));
      } else if (key == "max") {
        target->set_max_value(std::stod(value));
      } else if (key == "step") {
        target->set_step(std::stod(value));
      } else if (key == "min_items") {
        target->set_min_items(static_cast<uint32_t>(std::stoul(value)));
      } else if (key == "max_items") {
        target->set_max_items(static_cast<uint32_t>(std::stoul(value)));
      } else if (key == "pattern") {
        target->set_pattern(value);
      }
    } catch (const std::exception&) {
      // Keep metadata passthrough even if typed conversion fails.
    }
  }
}

}  // namespace jrb::application::server_info

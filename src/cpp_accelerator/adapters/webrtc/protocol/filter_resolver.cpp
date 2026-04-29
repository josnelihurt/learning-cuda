#include "src/cpp_accelerator/adapters/webrtc/protocol/filter_resolver.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace jrb::adapters::webrtc::protocol {

std::string NormalizeFilterId(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized;
}

std::optional<std::string> FirstGenericValue(
    const cuda_learning::GenericFilterParameterSelection& selection) {
  if (selection.values().empty()) return std::nullopt;
  return selection.values(0);
}

cuda_learning::GrayscaleType MapStringToGrayscaleType(const std::string& value) {
  const std::string normalized = NormalizeFilterId(value);
  if (normalized == "bt709")      return cuda_learning::GRAYSCALE_TYPE_BT709;
  if (normalized == "average")    return cuda_learning::GRAYSCALE_TYPE_AVERAGE;
  if (normalized == "lightness")  return cuda_learning::GRAYSCALE_TYPE_LIGHTNESS;
  if (normalized == "luminosity") return cuda_learning::GRAYSCALE_TYPE_LUMINOSITY;
  return cuda_learning::GRAYSCALE_TYPE_BT601;
}

cuda_learning::BorderMode MapStringToBorderMode(const std::string& value) {
  const std::string normalized = NormalizeFilterId(value);
  if (normalized == "clamp") return cuda_learning::BORDER_MODE_CLAMP;
  if (normalized == "wrap")  return cuda_learning::BORDER_MODE_WRAP;
  return cuda_learning::BORDER_MODE_REFLECT;
}

bool ParseBool(const std::string& value) {
  const std::string normalized = NormalizeFilterId(value);
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

void ResolveGenericSelectionsInPlace(cuda_learning::ProcessImageRequest* request) {
  if (request == nullptr || request->generic_filters_size() == 0) return;

  std::vector<cuda_learning::FilterType> filters;
  cuda_learning::GrayscaleType grayscale = request->grayscale_type();
  cuda_learning::GaussianBlurParameters blur_params = request->blur_params();
  bool has_blur_params = request->has_blur_params();

  for (const auto& selection : request->generic_filters()) {
    const std::string filter_id = NormalizeFilterId(selection.filter_id());
    if (filter_id.empty() || filter_id == "none") {
      filters.push_back(cuda_learning::FILTER_TYPE_NONE);
      continue;
    }

    if (filter_id == "grayscale") {
      filters.push_back(cuda_learning::FILTER_TYPE_GRAYSCALE);
      for (const auto& parameter : selection.parameters()) {
        if (NormalizeFilterId(parameter.parameter_id()) != "algorithm") continue;
        const auto value = FirstGenericValue(parameter);
        if (value.has_value()) {
          grayscale = MapStringToGrayscaleType(*value);
        }
      }
      continue;
    }

    if (filter_id == "blur") {
      filters.push_back(cuda_learning::FILTER_TYPE_BLUR);
      for (const auto& parameter : selection.parameters()) {
        const auto value = FirstGenericValue(parameter);
        if (!value.has_value()) continue;
        const std::string parameter_id = NormalizeFilterId(parameter.parameter_id());
        if (parameter_id == "kernel_size") {
          try {
            int parsed = std::stoi(*value);
            parsed = std::max(1, parsed);
            if (parsed % 2 == 0) parsed += 1;
            blur_params.set_kernel_size(parsed);
            has_blur_params = true;
          } catch (const std::exception&) {
            spdlog::warn("Ignoring invalid blur kernel_size value: {}", *value);
          }
          continue;
        }
        if (parameter_id == "sigma") {
          try {
            const float parsed = std::stof(*value);
            if (parsed >= 0.0F) {
              blur_params.set_sigma(parsed);
              has_blur_params = true;
            }
          } catch (const std::exception&) {
            spdlog::warn("Ignoring invalid blur sigma value: {}", *value);
          }
          continue;
        }
        if (parameter_id == "border_mode") {
          blur_params.set_border_mode(MapStringToBorderMode(*value));
          has_blur_params = true;
          continue;
        }
        if (parameter_id == "separable") {
          blur_params.set_separable(ParseBool(*value));
          has_blur_params = true;
        }
      }
      continue;
    }

    if (filter_id == "model_inference") {
      filters.push_back(cuda_learning::FILTER_TYPE_MODEL_INFERENCE);
      auto* model_params = request->mutable_model_params();
      if (model_params->model_id().empty()) {
        model_params->set_model_id("yolov10n");
      }
      if (model_params->confidence_threshold() <= 0.0F) {
        model_params->set_confidence_threshold(0.5F);
      }
      for (const auto& parameter : selection.parameters()) {
        const auto value = FirstGenericValue(parameter);
        if (!value.has_value()) continue;
        const std::string parameter_id = NormalizeFilterId(parameter.parameter_id());
        if (parameter_id == "model_id") {
          if (!value->empty()) model_params->set_model_id(*value);
        } else if (parameter_id == "confidence_threshold") {
          try {
            const float parsed = std::stof(*value);
            if (parsed > 0.0F) model_params->set_confidence_threshold(parsed);
          } catch (const std::exception&) {
            spdlog::warn("Ignoring invalid model confidence_threshold: {}", *value);
          }
        }
      }
      continue;
    }

    spdlog::warn("Ignoring unsupported generic filter in ProcessImageRequest: {}",
                 selection.filter_id());
  }

  request->clear_filters();
  for (const auto filter : filters) {
    request->add_filters(filter);
  }

  if (grayscale == cuda_learning::GRAYSCALE_TYPE_UNSPECIFIED) {
    grayscale = cuda_learning::GRAYSCALE_TYPE_BT601;
  }
  request->set_grayscale_type(grayscale);

  if (has_blur_params) {
    request->mutable_blur_params()->CopyFrom(blur_params);
  }
}

}  // namespace jrb::adapters::webrtc::protocol

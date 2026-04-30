#pragma once

#include <string>
#include <vector>

#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "src/cpp_accelerator/domain/interfaces/grayscale_algorithm.h"

namespace jrb::application::engine {

// ── Parameter descriptor ──────────────────────────────────────────────────────

struct ParameterOption {
  std::string value;
  std::string label;
};

struct ParameterDescriptor {
  std::string id;
  std::string name;
  std::string type;  // "select" | "range" | "number" | "checkbox"
  std::string default_value;
  std::vector<ParameterOption> options;
  // Validation metadata (min, max, step, required, min_items, max_items, etc.)
  std::vector<std::pair<std::string, std::string>> metadata;
};

// ── Filter descriptor (one per supported filter per factory) ──────────────────

struct FilterDescriptor {
  std::string id;
  std::string name;
  std::vector<ParameterDescriptor> parameters;
};

// ── Runtime creation parameters (factories read what they understand) ─────────

enum class BlurBorderMode : std::uint8_t { CLAMP, REFLECT, WRAP };

struct FilterCreationParams {
  // Grayscale
  jrb::domain::interfaces::GrayscaleAlgorithm grayscale_algorithm =
      jrb::domain::interfaces::GrayscaleAlgorithm::BT601;

  // Blur
  int blur_kernel_size = 5;
  float blur_sigma = 1.0F;
  BlurBorderMode blur_border_mode = BlurBorderMode::REFLECT;
  bool blur_separable = true;
};

}  // namespace jrb::application::engine

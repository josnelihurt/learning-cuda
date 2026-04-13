#pragma once

#include <cstdint>
#include <string>

namespace jrb::infrastructure::config::models {

enum class ProgramType : std::uint8_t {
  Passthrough,       // Renamed from Simple - no-op command
  CudaImageFilters,  // Renamed from Grayscale - GPU-accelerated filters
  CpuImageFilters    // New - CPU-based filters
};

struct ProgramConfig {
  std::string input_image_path;
  std::string output_image_path;
  ProgramType program_type;
};

}  // namespace jrb::infrastructure::config::models

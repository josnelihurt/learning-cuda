#pragma once

#include <string>
#include <cstdint>

namespace jrb::infrastructure::config::models {

enum class ProgramType : std::uint8_t {
    Simple,
    Grayscale
};

struct ProgramConfig {
    std::string input_image_path;
    std::string output_image_path;
    ProgramType program_type;
};

}  // namespace jrb::infrastructure::config::models

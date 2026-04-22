#pragma once

#include <string>

namespace jrb::infrastructure::cuda {

struct Detection {
    float x, y, width, height;
    int class_id;
    std::string class_name;
    float confidence;
};

}  // namespace jrb::infrastructure::cuda

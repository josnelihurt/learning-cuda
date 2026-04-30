#pragma once

#include <string>

namespace jrb::adapters::compute::cuda {

struct Detection {
    float x, y, width, height;
    int class_id;
    std::string class_name;
    float confidence;
};

}  // namespace jrb::adapters::compute::cuda

#pragma once

#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "src/cpp_accelerator/domain/models/detection.h"
#include <vector>

namespace jrb::infrastructure::cuda {

class IYoloDetector : public jrb::domain::interfaces::IFilter {
public:
    ~IYoloDetector() override = default;
    virtual const std::vector<Detection>& GetDetections() const = 0;
};

}  // namespace jrb::infrastructure::cuda

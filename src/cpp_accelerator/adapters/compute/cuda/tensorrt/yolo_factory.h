#pragma once

#include "src/cpp_accelerator/domain/interfaces/i_yolo_detector.h"
#include <memory>
#include <string>

namespace jrb::infrastructure::cuda {

std::shared_ptr<IYoloDetector> CreateYoloDetector(
    const std::string& model_path, float confidence_threshold);

}  // namespace jrb::infrastructure::cuda

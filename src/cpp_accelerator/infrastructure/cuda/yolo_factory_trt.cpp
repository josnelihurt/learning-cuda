#include "src/cpp_accelerator/infrastructure/cuda/yolo_factory.h"
#include "src/cpp_accelerator/infrastructure/cuda/yolo_detector.h"

namespace jrb::infrastructure::cuda {

std::shared_ptr<IYoloDetector> CreateYoloDetector(
    const std::string& model_path, float confidence_threshold) {
    return std::make_shared<YOLODetector>(model_path, confidence_threshold);
}

}  // namespace jrb::infrastructure::cuda

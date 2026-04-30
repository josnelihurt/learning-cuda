#include "src/cpp_accelerator/adapters/compute/cuda/tensorrt/yolo_factory.h"
#include "src/cpp_accelerator/adapters/compute/cuda/tensorrt/yolo_detector.h"

namespace jrb::adapters::compute::cuda {

std::shared_ptr<IYoloDetector> CreateYoloDetector(
    const std::string& model_path, float confidence_threshold) {
    return std::make_shared<YOLODetector>(model_path, confidence_threshold);
}

}  // namespace jrb::adapters::compute::cuda

#pragma once

#include "src/cpp_accelerator/infrastructure/cuda/i_yolo_detector.h"
#include <memory>
#include <string>

namespace jrb::infrastructure::cuda {

class YOLODetector : public IYoloDetector {
public:
    explicit YOLODetector(const std::string& model_path, float confidence_threshold = 0.5f);
    ~YOLODetector() override;

    bool Apply(jrb::domain::interfaces::FilterContext& context) override;
    jrb::domain::interfaces::FilterType GetType() const override;
    bool IsInPlace() const override;

    const std::vector<Detection>& GetDetections() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    float confidence_threshold_;
};

}  // namespace jrb::infrastructure::cuda

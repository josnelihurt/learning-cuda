#pragma once

#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include <memory>
#include <vector>
#include <string>

namespace jrb::infrastructure::cuda {

struct Detection {
    float x, y, width, height;
    int class_id;
    std::string class_name;
    float confidence;
};

class YOLODetector : public jrb::domain::interfaces::IFilter {
public:
    explicit YOLODetector(const std::string& model_path, float confidence_threshold = 0.5f);
    ~YOLODetector() override;

    bool Apply(jrb::domain::interfaces::FilterContext& context) override;
    jrb::domain::interfaces::FilterType GetType() const override;
    bool IsInPlace() const override;

    const std::vector<Detection>& GetDetections() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    float confidence_threshold_;
    std::vector<Detection> detections_;
};

}  // namespace jrb::infrastructure::cuda

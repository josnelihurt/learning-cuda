#include "src/cpp_accelerator/infrastructure/cuda/yolo_detector_stub.h"
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace jrb::infrastructure::cuda {

YOLODetectorStub::YOLODetectorStub(const std::string& model_path, float confidence_threshold)
    : confidence_threshold_(confidence_threshold) {
    spdlog::warn("YOLODetector stub used - YOLO inference not available on this platform. Model: {}", model_path);
    spdlog::warn("This is expected on ARM64 CI without TensorRT. Use TensorRT-enabled build for actual inference.");
}

YOLODetectorStub::~YOLODetectorStub() = default;

bool YOLODetectorStub::Apply(jrb::domain::interfaces::FilterContext& context) {
    // Stub: pass through the image without running inference
    if (context.input.data != context.output.data) {
        const int width = context.input.width;
        const int height = context.input.height;
        const int in_channels = context.input.channels;
        const int out_channels = context.output.channels;
        const size_t data_size = static_cast<size_t>(width) * height;

        if (in_channels == out_channels) {
            std::memcpy(context.output.data, context.input.data, data_size * in_channels);
        } else {
            // Handle channel mismatch
            for (size_t i = 0; i < data_size; ++i) {
                const uint8_t* src = context.input.data + i * in_channels;
                uint8_t* dst = context.output.data + i * out_channels;
                for (int c = 0; c < out_channels; ++c) {
                    dst[c] = (c < in_channels) ? src[c] : 0xFF;
                }
            }
        }
    }

    detections_.clear();
    spdlog::debug("YOLODetector stub: no detections (inference not available)");
    return true;
}

jrb::domain::interfaces::FilterType YOLODetectorStub::GetType() const {
    return jrb::domain::interfaces::FilterType::MODEL_INFERENCE;
}

bool YOLODetectorStub::IsInPlace() const {
    return true;
}

const std::vector<Detection>& YOLODetectorStub::GetDetections() const {
    return detections_;
}

}  // namespace jrb::infrastructure::cuda

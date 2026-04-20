#include "src/cpp_accelerator/infrastructure/cuda/yolo_detector.h"
#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <algorithm>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::infrastructure::cuda {

class YOLODetector::Impl {
public:
    Impl(const std::string& model_path, float confidence_threshold)
        : confidence_threshold_(confidence_threshold) {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLODetector");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    }

    std::vector<Detection> Detect(const uint8_t* image_data, int width, int height, int channels) {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        size_t input_tensor_size = 1 * 3 * 640 * 640;
        std::vector<float> input_tensor_values(input_tensor_size);

        Preprocess(image_data, width, height, channels, input_tensor_values.data(), 640, 640);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_shape.data(), 4));

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_, input_tensors.data(), 1,
            output_names_, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        return Postprocess(output_data, output_shape);
    }

private:
    void Preprocess(const uint8_t* src, int src_w, int src_h, int src_c,
                    float* dst, int dst_w, int dst_h) {
        float scale_x = static_cast<float>(dst_w) / src_w;
        float scale_y = static_cast<float>(dst_h) / src_h;
        float scale = std::min(scale_x, scale_y);

        int pad_w = (dst_w - static_cast<int>(src_w * scale)) / 2;
        int pad_h = (dst_h - static_cast<int>(src_h * scale)) / 2;

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < dst_h; ++y) {
                for (int x = 0; x < dst_w; ++x) {
                    int src_x = static_cast<int>((x - pad_w) / scale);
                    int src_y = static_cast<int>((y - pad_h) / scale);

                    float value = 0.0f;
                    if (src_x >= 0 && src_x < src_w && src_y >= 0 && src_y < src_h) {
                        int actual_src_c = (src_c == 4) ? (c == 2 ? 2 : c + 1) : c;
                        value = static_cast<float>(src[(src_y * src_w + src_x) * src_c + actual_src_c]) / 255.0f;
                    }

                    dst[c * dst_w * dst_h + y * dst_w + x] = value;
                }
            }
        }
    }

    std::vector<Detection> Postprocess(float* output, const std::vector<int64_t>& shape) {
        std::vector<Detection> detections;

        int num_classes = shape[2] - 4;
        int num_predictions = shape[1];

        for (int i = 0; i < num_predictions; ++i) {
            float* prediction = output + i * (4 + num_classes);

            float x_center = prediction[0];
            float y_center = prediction[1];
            float width = prediction[2];
            float height = prediction[3];

            float max_confidence = 0.0f;
            int max_class_id = 0;

            for (int c = 0; c < num_classes; ++c) {
                float conf = prediction[4 + c];
                if (conf > max_confidence) {
                    max_confidence = conf;
                    max_class_id = c;
                }
            }

            if (max_confidence >= confidence_threshold_) {
                detections.push_back({
                    .x = x_center - width / 2.0f,
                    .y = y_center - height / 2.0f,
                    .width = width,
                    .height = height,
                    .class_id = max_class_id,
                    .confidence = max_confidence
                });
            }
        }

        return detections;
    }

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    float confidence_threshold_;

    static constexpr const char* input_names_[] = {"images"};
    static constexpr const char* output_names_[] = {"output0"};
};

YOLODetector::YOLODetector(const std::string& model_path, float confidence_threshold)
    : impl_(std::make_unique<Impl>(model_path, confidence_threshold))
    , confidence_threshold_(confidence_threshold) {
    spdlog::info("YOLODetector initialized with model: {}", model_path);
}

YOLODetector::~YOLODetector() = default;

bool YOLODetector::Apply(jrb::domain::interfaces::FilterContext& context) {
    detections_ = impl_->Detect(
        context.input.data,
        context.input.width,
        context.input.height,
        context.input.channels
    );

    spdlog::info("YOLO detected {} objects", detections_.size());
    return true;
}

jrb::domain::interfaces::FilterType YOLODetector::GetType() const {
    return jrb::domain::interfaces::FilterType::GRAYSCALE;
}

bool YOLODetector::IsInPlace() const {
    return true;
}

const std::vector<Detection>& YOLODetector::GetDetections() const {
    return detections_;
}

}  // namespace jrb::infrastructure::cuda

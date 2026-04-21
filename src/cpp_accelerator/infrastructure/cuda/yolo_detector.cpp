#include "src/cpp_accelerator/infrastructure/cuda/yolo_detector.h"
#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <algorithm>
#include <cstring>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

// COCO 80 class names (indices 0-79)
static constexpr const char* kCocoClassNames[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};
static_assert(sizeof(kCocoClassNames) / sizeof(kCocoClassNames[0]) == 80,
              "kCocoClassNames must have exactly 80 entries");

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

        // YOLOv10 end-to-end NMS: shape [1, N, 6] = (x1, y1, x2, y2, conf, class_id)
        if (shape.size() == 3 && shape[2] == 6) {
            const int num_predictions = static_cast<int>(shape[1]);
            for (int i = 0; i < num_predictions; ++i) {
                float* p = output + i * 6;
                float x1 = p[0];
                float y1 = p[1];
                float x2 = p[2];
                float y2 = p[3];
                float conf = p[4];
                int class_id = static_cast<int>(p[5]);

                if (conf < confidence_threshold_) {
                    continue;
                }
                if (x2 <= x1 || y2 <= y1) {
                    continue;
                }
                detections.push_back({
                    .x = x1,
                    .y = y1,
                    .width = x2 - x1,
                    .height = y2 - y1,
                    .class_id = class_id,
                    .class_name = kCocoClassNames[std::min(class_id, 79)],
                    .confidence = conf,
                });
            }
            return detections;
        }

        // YOLOv8-style: [1, N, 4 + num_classes] = (cx, cy, w, h, class0_conf, class1_conf, ...)
        int num_classes = static_cast<int>(shape[2]) - 4;
        int num_predictions = static_cast<int>(shape[1]);

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
                    .class_name = kCocoClassNames[std::min(max_class_id, 79)],
                    .confidence = max_confidence,
                });
            }
        }

        return ApplyNMS(detections, 0.45f);
    }

    static float BoxIoU(const Detection& a, const Detection& b) {
        float ax2 = a.x + a.width;
        float ay2 = a.y + a.height;
        float bx2 = b.x + b.width;
        float by2 = b.y + b.height;
        float ix1 = std::max(a.x, b.x);
        float iy1 = std::max(a.y, b.y);
        float ix2 = std::min(ax2, bx2);
        float iy2 = std::min(ay2, by2);
        float iw = std::max(0.0f, ix2 - ix1);
        float ih = std::max(0.0f, iy2 - iy1);
        float inter = iw * ih;
        float area_a = a.width * a.height;
        float area_b = b.width * b.height;
        float denom = area_a + area_b - inter;
        return denom > 0.0f ? inter / denom : 0.0f;
    }

    static std::vector<Detection> ApplyNMS(std::vector<Detection> dets, float iou_threshold) {
        std::sort(dets.begin(), dets.end(),
                  [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
        std::vector<Detection> kept;
        std::vector<bool> suppressed(dets.size(), false);
        for (size_t i = 0; i < dets.size(); ++i) {
            if (suppressed[i]) continue;
            kept.push_back(dets[i]);
            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (suppressed[j]) continue;
                if (dets[j].class_id != dets[i].class_id) continue;
                if (BoxIoU(dets[i], dets[j]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        return kept;
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

namespace {

void CopyInputToOutput(const jrb::domain::interfaces::FilterContext& context) {
    const int width = context.input.width;
    const int height = context.input.height;
    const int in_channels = context.input.channels;
    const int out_channels = context.output.channels;

    if (in_channels == out_channels) {
        std::memcpy(context.output.data, context.input.data,
                    static_cast<size_t>(width) * height * in_channels);
        return;
    }

    const int pixels = width * height;
    for (int i = 0; i < pixels; ++i) {
        const uint8_t* src = context.input.data + i * in_channels;
        uint8_t* dst = context.output.data + i * out_channels;
        for (int c = 0; c < out_channels; ++c) {
            dst[c] = (c < in_channels) ? src[c] : 0xFF;
        }
    }
}

}  // namespace

bool YOLODetector::Apply(jrb::domain::interfaces::FilterContext& context) {
    const int width = context.input.width;
    const int height = context.input.height;

    auto raw = impl_->Detect(
        context.input.data, width, height, context.input.channels
    );

    // Convert from 640x640 model space to original image pixel space
    constexpr int kModelSize = 640;
    const float scale = std::min(static_cast<float>(kModelSize) / width,
                                 static_cast<float>(kModelSize) / height);
    const float pad_x = (kModelSize - width * scale) * 0.5f;
    const float pad_y = (kModelSize - height * scale) * 0.5f;

    detections_.clear();
    for (const auto& det : raw) {
        float x = std::max(0.0f, (det.x - pad_x) / scale);
        float y = std::max(0.0f, (det.y - pad_y) / scale);
        float w = std::min(static_cast<float>(width) - x, det.width / scale);
        float h = std::min(static_cast<float>(height) - y, det.height / scale);
        if (w <= 0.0f || h <= 0.0f) continue;
        detections_.push_back({x, y, w, h, det.class_id, det.class_name, det.confidence});
    }

    CopyInputToOutput(context);

    spdlog::info("YOLO detected {} objects", detections_.size());
    return true;
}

jrb::domain::interfaces::FilterType YOLODetector::GetType() const {
    return jrb::domain::interfaces::FilterType::MODEL_INFERENCE;
}

bool YOLODetector::IsInPlace() const {
    return true;
}

const std::vector<Detection>& YOLODetector::GetDetections() const {
    return detections_;
}

}  // namespace jrb::infrastructure::cuda

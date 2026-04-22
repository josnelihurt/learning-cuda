#include "src/cpp_accelerator/infrastructure/cuda/yolo_detector.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "src/cpp_accelerator/infrastructure/cuda/letterbox_kernel.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

// COCO 80 class names (indices 0-79)
static constexpr const char* kCocoClassNames[] = {"person",        "bicycle",      "car",
                                                  "motorcycle",    "airplane",     "bus",
                                                  "train",         "truck",        "boat",
                                                  "traffic light", "fire hydrant", "stop sign",
                                                  "parking meter", "bench",        "bird",
                                                  "cat",           "dog",          "horse",
                                                  "sheep",         "cow",          "elephant",
                                                  "bear",          "zebra",        "giraffe",
                                                  "backpack",      "umbrella",     "handbag",
                                                  "tie",           "suitcase",     "frisbee",
                                                  "skis",          "snowboard",    "sports ball",
                                                  "kite",          "baseball bat", "baseball glove",
                                                  "skateboard",    "surfboard",    "tennis racket",
                                                  "bottle",        "wine glass",   "cup",
                                                  "fork",          "knife",        "spoon",
                                                  "bowl",          "banana",       "apple",
                                                  "sandwich",      "orange",       "broccoli",
                                                  "carrot",        "hot dog",      "pizza",
                                                  "donut",         "cake",         "chair",
                                                  "couch",         "potted plant", "bed",
                                                  "dining table",  "toilet",       "tv",
                                                  "laptop",        "mouse",        "remote",
                                                  "keyboard",      "cell phone",   "microwave",
                                                  "oven",          "toaster",      "sink",
                                                  "refrigerator",  "book",         "clock",
                                                  "vase",          "scissors",     "teddy bear",
                                                  "hair drier",    "toothbrush"};
static_assert(sizeof(kCocoClassNames) / sizeof(kCocoClassNames[0]) == 80,
              "kCocoClassNames must have exactly 80 entries");

namespace jrb::infrastructure::cuda {

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        spdlog::error("[TRT] {}", msg);
        break;
      case Severity::kWARNING:
        spdlog::warn("[TRT] {}", msg);
        break;
      case Severity::kINFO:
        spdlog::info("[TRT] {}", msg);
        break;
      case Severity::kVERBOSE:
        spdlog::debug("[TRT] {}", msg);
        break;
    }
  }
};

class YOLODetector::Impl {
public:
  Impl(const std::string& model_path, float confidence_threshold)
      : logger_(std::make_unique<TRTLogger>()),
        confidence_threshold_(confidence_threshold),
        model_path_(model_path),
        input_tensor_values_(1 * 3 * 640 * 640) {
    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (!runtime_) {
      throw std::runtime_error("Failed to create TensorRT runtime");
    }

    LoadEngine();
  }

  ~Impl() {
    // TensorRT 10.x: objects are managed by the library, no manual destroy needed
    // The runtime will clean up associated resources
  }

  std::vector<Detection> Detect(const uint8_t* image_data, int width, int height, int channels) {
    cudaError_t cuda_err = cuda_letterbox_resize(image_data, width, height, channels,
                                                 input_tensor_values_.data(), 640, 640);
    if (cuda_err != cudaSuccess) {
      spdlog::error("cuda_letterbox_resize failed: {}", cudaGetErrorString(cuda_err));
      return {};
    }

    if (!RunInference(input_tensor_values_.data())) {
      return {};
    }

    return Postprocess(output_buffer_.data(), output_shape_);
  }

private:
  void LoadEngine() {
    std::string engine_path = GetEnginePath();

    // Try to load cached engine
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (engine_file.good()) {
      spdlog::info("Loading cached TensorRT engine from: {}", engine_path);
      engine_file.seekg(0, std::ios::end);
      size_t engine_size = engine_file.tellg();
      engine_file.seekg(0, std::ios::beg);
      std::vector<char> engine_data(engine_size);
      engine_file.read(engine_data.data(), engine_size);
      engine_file.close();

      engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_size);

      if (engine_) {
        spdlog::info("Successfully loaded cached engine");
        context_ = engine_->createExecutionContext();
        if (!context_) {
          throw std::runtime_error("Failed to create execution context");
        }
        PrepareBuffers();
        return;
      }
    }

    // No cached engine or load failed, build from ONNX
    spdlog::info("No cached engine found, building from ONNX: {}", model_path_);
    BuildEngineFromOnnx();

    // Save engine for future runs
    SaveEngine(engine_path);
  }

  std::string GetEnginePath() {
    std::string base_path = model_path_;
    // Remove .onnx extension if present
    if (base_path.length() > 5 && base_path.substr(base_path.length() - 5) == ".onnx") {
      base_path = base_path.substr(0, base_path.length() - 5);
    }

    #if defined(__aarch64__)
      // Try JetPack-specific engines first
      std::string jp46_path = base_path + ".jp46.engine";
      if (std::ifstream(jp46_path).good()) {
        return jp46_path;
      }
      std::string jp6_path = base_path + ".jp6.engine";
      if (std::ifstream(jp6_path).good()) {
        return jp6_path;
      }
    #endif

    return base_path + ".engine";
  }

  void BuildEngineFromOnnx() {
    auto builder = nvinfer1::createInferBuilder(*logger_);
    if (!builder) {
      throw std::runtime_error("Failed to create TensorRT builder");
    }

    // Use 0U instead of deprecated kEXPLICIT_BATCH for TRT 10.x
    auto network = builder->createNetworkV2(0U);
    if (!network) {
      throw std::runtime_error("Failed to create network definition");
    }

    auto config = builder->createBuilderConfig();
    if (!config) {
      throw std::runtime_error("Failed to create builder config");
    }

    auto parser = nvonnxparser::createParser(*network, *logger_);
    if (!parser) {
      throw std::runtime_error("Failed to create ONNX parser");
    }

    // Parse ONNX model
    if (!parser->parseFromFile(model_path_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
      std::string errors;
      for (int i = 0; i < parser->getNbErrors(); ++i) {
        auto error = parser->getError(i);
        errors += std::string(error->desc()) + "\n";
      }
      throw std::runtime_error("Failed to parse ONNX model: " + errors);
    }

    // Build engine
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB

    // Build engine (using buildEngineWithConfig for TRT 10.x compatibility)
    // Note: builder, network, config, parser are managed by TensorRT library
    engine_ = builder->buildEngineWithConfig(*network, *config);
    if (!engine_) {
      throw std::runtime_error("Failed to build TensorRT engine");
    }

    spdlog::info("Successfully built TensorRT engine from ONNX");

    context_ = engine_->createExecutionContext();
    if (!context_) {
      throw std::runtime_error("Failed to create execution context");
    }

    PrepareBuffers();
  }

  void PrepareBuffers() {
    // Get input/output info
    int num_io = engine_->getNbIOTensors();
    for (int i = 0; i < num_io; ++i) {
      const char* name = engine_->getIOTensorName(i);
      nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);

      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        input_name_ = name;
        auto dims = engine_->getTensorShape(name);
        input_size_ = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
          input_size_ *= dims.d[j];
        }
        spdlog::debug("Input: {} size={}", name, input_size_);
      } else {
        output_name_ = name;
        auto dims = engine_->getTensorShape(name);
        output_size_ = 1;
        output_shape_.clear();
        for (int j = 0; j < dims.nbDims; ++j) {
          output_size_ *= dims.d[j];
          output_shape_.push_back(dims.d[j]);
        }
        // Build shape string for debug output
        std::string shape_str;
        for (size_t j = 0; j < output_shape_.size(); ++j) {
          if (j > 0) shape_str += ",";
          shape_str += std::to_string(output_shape_[j]);
        }
        spdlog::debug("Output: {} size={} shape=[{}]", name, output_size_, shape_str);
      }
    }

    output_buffer_.resize(output_size_);
  }

  bool RunInference(float* input_data) {
    if (!context_ || !engine_) {
      spdlog::error("Engine or context not initialized");
      return false;
    }

    // For simplicity, we're using host memory for now.
    // TODO: Use CUDA device memory directly to avoid host round-trip
    void* buffers[2] = {input_data, output_buffer_.data()};

    // Set tensor addresses
    if (!context_->setTensorAddress(input_name_.c_str(), buffers[0])) {
      spdlog::error("Failed to set input tensor address");
      return false;
    }
    if (!context_->setTensorAddress(output_name_.c_str(), buffers[1])) {
      spdlog::error("Failed to set output tensor address");
      return false;
    }

    // Execute inference
    if (!context_->enqueueV3(0)) {  // 0 = default CUDA stream
      spdlog::error("TensorRT inference failed");
      return false;
    }

    // Synchronize (since we're using host memory)
    cudaDeviceSynchronize();

    return true;
  }

  void SaveEngine(const std::string& path) {
    nvinfer1::IHostMemory* serialized = engine_->serialize();
    if (!serialized) {
      spdlog::error("Failed to serialize engine");
      return;
    }

    std::ofstream engine_file(path, std::ios::binary);
    if (!engine_file) {
      spdlog::error("Failed to open engine file for writing: {}", path);
      return;
    }

    engine_file.write(static_cast<const char*>(serialized->data()), serialized->size());
    engine_file.close();
    spdlog::info("Saved engine to: {}", path);
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
      if (suppressed[i])
        continue;
      kept.push_back(dets[i]);
      for (size_t j = i + 1; j < dets.size(); ++j) {
        if (suppressed[j])
          continue;
        if (dets[j].class_id != dets[i].class_id)
          continue;
        if (BoxIoU(dets[i], dets[j]) > iou_threshold) {
          suppressed[j] = true;
        }
      }
    }
    return kept;
  }

  std::unique_ptr<TRTLogger> logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  float confidence_threshold_;
  std::string model_path_;
  std::vector<float> input_tensor_values_;
  std::vector<float> output_buffer_;

  std::string input_name_;
  std::string output_name_;
  size_t input_size_ = 0;
  size_t output_size_ = 0;
  std::vector<int64_t> output_shape_;
};

YOLODetector::YOLODetector(const std::string& model_path, float confidence_threshold)
    : impl_(std::make_unique<Impl>(model_path, confidence_threshold)),
      confidence_threshold_(confidence_threshold) {
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

  auto raw = impl_->Detect(context.input.data, width, height, context.input.channels);

  // Convert from 640x640 model space to original image pixel space
  constexpr int kModelSize = 640;
  const float scale =
      std::min(static_cast<float>(kModelSize) / width, static_cast<float>(kModelSize) / height);
  const float pad_x = (kModelSize - width * scale) * 0.5f;
  const float pad_y = (kModelSize - height * scale) * 0.5f;

  detections_.clear();
  for (const auto& det : raw) {
    float x = std::max(0.0f, (det.x - pad_x) / scale);
    float y = std::max(0.0f, (det.y - pad_y) / scale);
    float w = std::min(static_cast<float>(width) - x, det.width / scale);
    float h = std::min(static_cast<float>(height) - y, det.height / scale);
    if (w <= 0.0f || h <= 0.0f)
      continue;
    detections_.push_back({x, y, w, h, det.class_id, det.class_name, det.confidence});
  }

  CopyInputToOutput(context);

  spdlog::debug("YOLO detected {} objects", detections_.size());
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

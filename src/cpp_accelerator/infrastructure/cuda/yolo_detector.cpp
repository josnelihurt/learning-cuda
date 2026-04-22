#include "src/cpp_accelerator/infrastructure/cuda/yolo_detector.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
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
        model_path_(model_path) {
    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (!runtime_) {
      throw std::runtime_error("Failed to create TensorRT runtime");
    }

    LoadEngine();
  }

  ~Impl() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
  }

  std::vector<Detection> Detect(const uint8_t* image_data, int width, int height, int channels) {
    // Serialize access to GPU buffers and TRT context — not thread-safe.
    std::lock_guard<std::mutex> lock(inference_mutex_);

    cudaError_t cuda_err = cuda_letterbox_resize_to_device(image_data, width, height, channels,
                                                           d_input_, 640, 640);
    if (cuda_err != cudaSuccess) {
      spdlog::error("cuda_letterbox_resize_to_device failed: {}", cudaGetErrorString(cuda_err));
      return {};
    }

    if (!RunInference()) {
      return {};
    }

    // Copy output from GPU to host for postprocessing
    cudaError_t copy_err = cudaMemcpy(h_output_.data(), d_output_,
                                      output_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_err != cudaSuccess) {
      spdlog::error("Output D2H copy failed: {}", cudaGetErrorString(copy_err));
      return {};
    }

    return Postprocess(h_output_.data(), output_shape_);
  }

private:
  void LoadEngine() {
    std::string engine_path = GetEnginePath();

    // Try to load cached engine
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (engine_file.good()) {
      spdlog::info("[TRT] Loading cached engine from: {} ...", engine_path);
      auto t0 = std::chrono::steady_clock::now();
      engine_file.seekg(0, std::ios::end);
      size_t engine_size = engine_file.tellg();
      engine_file.seekg(0, std::ios::beg);
      std::vector<char> engine_data(engine_size);
      engine_file.read(engine_data.data(), engine_size);
      engine_file.close();

      engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_size);

      if (engine_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        spdlog::info("[TRT] Cached engine loaded in {}ms", elapsed);
        context_ = engine_->createExecutionContext();
        if (!context_) {
          throw std::runtime_error("Failed to create execution context");
        }
        PrepareBuffers();
        return;
      }
      spdlog::warn("[TRT] Cached engine deserialization failed — rebuilding from ONNX");
    }

    // No cached engine or load failed, build from ONNX
    spdlog::info("[TRT] No cached engine found, building from ONNX: {}", model_path_);
    spdlog::info("[TRT] *** First-time engine build — this takes ~60-90s on x86, ~120s on Jetson ***");
    BuildEngineFromOnnx(engine_path);
  }

  std::string GetEnginePath() {
    std::string base_path = model_path_;
    // Remove .onnx extension if present
    if (base_path.length() > 5 && base_path.substr(base_path.length() - 5) == ".onnx") {
      base_path = base_path.substr(0, base_path.length() - 5);
    }

    #if defined(__aarch64__)
      // Jetson Orin (JetPack 6, TRT 10.x) engine cache
      std::string jp6_path = base_path + ".jp6.engine";
      if (std::ifstream(jp6_path).good()) {
        return jp6_path;
      }
    #endif

    return base_path + ".engine";
  }

  void BuildEngineFromOnnx(const std::string& engine_path) {
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

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB

    // TRT requires an optimization profile for any dynamic-shape inputs.
    // YOLOv10 exports with a dynamic batch dimension; fix it at 1 × 3 × 640 × 640.
    auto profile = builder->createOptimizationProfile();
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto* input = network->getInput(i);
      nvinfer1::Dims dims = input->getDimensions();
      // Replace any dynamic (-1) dimension with its fixed value.
      // Batch is always 1; spatial dims are always 640×640.
      nvinfer1::Dims fixed = dims;
      for (int d = 0; d < fixed.nbDims; ++d) {
        if (fixed.d[d] < 0) {
          fixed.d[d] = (d == 0) ? 1 : 640; // batch=1, spatial=640
        }
      }
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, fixed);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, fixed);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, fixed);
    }
    config->addOptimizationProfile(profile);

    spdlog::info("[TRT] Starting engine compilation (see TRT logs for per-layer progress)...");
    auto build_t0 = std::chrono::steady_clock::now();

    auto plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
      throw std::runtime_error("Failed to build TensorRT serialized engine");
    }
    auto build_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - build_t0)
                             .count();
    spdlog::info("[TRT] Engine compilation finished in {}s", build_elapsed);

    engine_ = runtime_->deserializeCudaEngine(plan->data(), plan->size());
    if (!engine_) {
      throw std::runtime_error("Failed to build TensorRT engine");
    }

    spdlog::info("Successfully built TensorRT engine from ONNX");
    context_ = engine_->createExecutionContext();
    if (!context_) {
      throw std::runtime_error("Failed to create execution context");
    }

    PrepareBuffers();
    SaveEngine(engine_path, plan);
  }

  void PrepareBuffers() {
    int num_io = engine_->getNbIOTensors();
    for (int i = 0; i < num_io; ++i) {
      const char* name = engine_->getIOTensorName(i);
      nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);

      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        input_name_ = name;
        auto dims = engine_->getTensorShape(name);
        nvinfer1::Dims fixed = dims;
        input_size_ = 1;
        for (int j = 0; j < fixed.nbDims; ++j) {
          if (fixed.d[j] < 0) {
            fixed.d[j] = (j == 0) ? 1 : 640;  // resolve any remaining dynamic dim
          }
          input_size_ *= fixed.d[j];
        }
        // Bind the fixed shape to the execution context so enqueueV3 knows tensor sizes
        context_->setInputShape(name, fixed);
        spdlog::info("[TRT] Input tensor '{}': {} floats", name, input_size_);
      } else {
        output_name_ = name;
        auto dims = context_->getTensorShape(name);  // context gives resolved shape
        output_size_ = 1;
        output_shape_.clear();
        for (int j = 0; j < dims.nbDims; ++j) {
          output_size_ *= dims.d[j];
          output_shape_.push_back(dims.d[j]);
        }
        std::string shape_str;
        for (size_t j = 0; j < output_shape_.size(); ++j) {
          if (j > 0) shape_str += ",";
          shape_str += std::to_string(output_shape_[j]);
        }
        spdlog::info("[TRT] Output tensor '{}': {} floats, shape=[{}]", name, output_size_,
                     shape_str);
      }
    }

    if (input_size_ == 0 || output_size_ == 0) {
      throw std::runtime_error("PrepareBuffers: zero-size tensor — engine may be malformed");
    }

    // Allocate persistent GPU buffers for TensorRT (setTensorAddress needs device pointers)
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_input_), input_size_ * sizeof(float));
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc input failed: ") + cudaGetErrorString(err));
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_output_), output_size_ * sizeof(float));
    if (err != cudaSuccess) {
      cudaFree(d_input_);
      d_input_ = nullptr;
      throw std::runtime_error(std::string("cudaMalloc output failed: ") +
                               cudaGetErrorString(err));
    }
    h_output_.resize(output_size_);
    spdlog::info("[TRT] GPU buffers allocated: input={}B output={}B",
                 input_size_ * sizeof(float), output_size_ * sizeof(float));
  }

  bool RunInference() {
    if (!context_ || !engine_) {
      spdlog::error("Engine or context not initialized");
      return false;
    }

    if (!context_->setTensorAddress(input_name_.c_str(), d_input_)) {
      spdlog::error("Failed to set input tensor address");
      return false;
    }
    if (!context_->setTensorAddress(output_name_.c_str(), d_output_)) {
      spdlog::error("Failed to set output tensor address");
      return false;
    }
    if (!context_->enqueueV3(0)) {
      spdlog::error("TensorRT inference failed");
      return false;
    }

    cudaDeviceSynchronize();
    return true;
  }

  void SaveEngine(const std::string& path, const nvinfer1::IHostMemory* plan) {
    std::ofstream engine_file(path, std::ios::binary);
    if (!engine_file) {
      spdlog::error("Failed to open engine file for writing: {}", path);
      return;
    }

    engine_file.write(static_cast<const char*>(plan->data()), plan->size());
    engine_file.close();
    spdlog::info("Saved engine to: {}", path);
  }

  std::vector<Detection> Postprocess(float* output, const std::vector<int64_t>& shape) {
    std::vector<Detection> detections;

    // YOLOv10 end-to-end NMS: shape [1, N, 6] = (x1, y1, x2, y2, conf, class_id)
    if (shape.size() == 3 && shape[2] == 6) {
      const int num_predictions = static_cast<int>(shape[1]);
      float best_conf = 0.0f;
      int best_class = -1;
      for (int i = 0; i < num_predictions; ++i) {
        float* p = output + i * 6;
        float x1 = p[0];
        float y1 = p[1];
        float x2 = p[2];
        float y2 = p[3];
        float conf = p[4];
        int class_id = static_cast<int>(p[5]);

        if (conf > best_conf) {
          best_conf = conf;
          best_class = class_id;
        }

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
      if (detections.empty() && best_conf > 0.0f) {
        spdlog::info("YOLO: best raw score was {:.3f} ({}) — below threshold {:.2f}",
                     best_conf, kCocoClassNames[std::min(best_class, 79)], confidence_threshold_);
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
  float* d_input_ = nullptr;   // GPU input buffer (1×3×640×640 floats)
  float* d_output_ = nullptr;  // GPU output buffer
  std::vector<float> h_output_; // CPU copy for postprocessing

  std::string input_name_;
  std::string output_name_;
  size_t input_size_ = 0;
  size_t output_size_ = 0;
  std::vector<int64_t> output_shape_;
  mutable std::mutex inference_mutex_;
};

YOLODetector::YOLODetector(const std::string& model_path, float confidence_threshold)
    : impl_(std::make_unique<Impl>(model_path, confidence_threshold)),
      confidence_threshold_(confidence_threshold) {
  spdlog::info("YOLODetector initialized with model: {}", model_path);
}

YOLODetector::~YOLODetector() = default;

namespace {

// Thread-local storage for detection results — each WebRTC callback thread gets its own copy,
// eliminating races between concurrent frame-processing threads.
thread_local std::vector<jrb::infrastructure::cuda::Detection> tl_detections;

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

  tl_detections.clear();
  for (const auto& det : raw) {
    float x = std::max(0.0f, (det.x - pad_x) / scale);
    float y = std::max(0.0f, (det.y - pad_y) / scale);
    float w = std::min(static_cast<float>(width) - x, det.width / scale);
    float h = std::min(static_cast<float>(height) - y, det.height / scale);
    if (w <= 0.0f || h <= 0.0f)
      continue;
    tl_detections.push_back({x, y, w, h, det.class_id, det.class_name, det.confidence});
  }

  CopyInputToOutput(context);

  spdlog::debug("YOLO raw: {} object(s) before coordinate transform", tl_detections.size());
  return true;
}

jrb::domain::interfaces::FilterType YOLODetector::GetType() const {
  return jrb::domain::interfaces::FilterType::MODEL_INFERENCE;
}

bool YOLODetector::IsInPlace() const {
  return true;
}

const std::vector<Detection>& YOLODetector::GetDetections() const {
  return tl_detections;
}

}  // namespace jrb::infrastructure::cuda

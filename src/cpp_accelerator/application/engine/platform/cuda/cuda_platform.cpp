#include "src/cpp_accelerator/application/engine/platform/cuda/cuda_platform.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/adapters/compute/cuda/cuda_filter_factory.h"
#include "src/cpp_accelerator/adapters/compute/cuda/tensorrt/model_manager.h"
#include "src/cpp_accelerator/adapters/compute/cuda/tensorrt/model_registry.h"
#include "src/cpp_accelerator/domain/interfaces/i_yolo_detector.h"

namespace jrb::application::engine::platform::cuda {

namespace {

std::unordered_map<std::string, std::shared_ptr<jrb::infrastructure::cuda::IYoloDetector>>
    g_detector_cache;

}  // namespace

void RegisterFactories(FilterFactoryRegistry& registry) {
  registry.Register(std::make_unique<jrb::infrastructure::cuda::CudaFilterFactory>());
}

void Initialize(const cuda_learning::InitRequest& /*request*/,
                cuda_learning::InitResponse* /*response*/) {
  auto& model_manager = jrb::infrastructure::cuda::ModelManager::GetInstance();
  jrb::infrastructure::cuda::ModelRegistry registry;
  registry.RegisterModel(
      {"yolov10n", "YOLO v10 Nano", "data/models/yolov10n.onnx", "Fastest YOLO v10 model"});
  model_manager.Initialize(registry);
  spdlog::info("Model manager initialized with {} models",
               model_manager.GetAvailableModels().size());

  auto available_models = model_manager.GetAvailableModels();
  spdlog::info("[Startup] Pre-loading {} TRT detector(s)...", available_models.size());
  for (const auto& model_id : available_models) {
    std::string cache_key = model_id + "@" + std::to_string(0.5f);
    auto t0 = std::chrono::steady_clock::now();
    auto detector = model_manager.GetDetector(model_id, 0.5f);
    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0)
            .count();
    if (detector) {
      g_detector_cache[cache_key] = std::move(detector);
      spdlog::info("[Startup] TRT engine ready for '{}' ({}s)", model_id, elapsed);
    } else {
      spdlog::error("[Startup] Failed to load TRT engine for '{}' after {}s", model_id, elapsed);
    }
  }
  spdlog::info("[Startup] All detectors loaded — ready to accept connections");
}

bool ApplyInference(const std::string& model_id, float confidence, bool pipeline_has_output,
                    const cuda_learning::ProcessImageRequest& /*request*/,
                    cuda_learning::ProcessImageResponse* response,
                    const jrb::domain::interfaces::ImageBuffer& input_buffer,
                    jrb::domain::interfaces::ImageBufferMut& output_buffer) {
  std::string cache_key = model_id + "@" + std::to_string(confidence);
  auto cache_it = g_detector_cache.find(cache_key);
  if (cache_it == g_detector_cache.end()) {
    auto new_detector =
        jrb::infrastructure::cuda::ModelManager::GetInstance().GetDetector(model_id, confidence);
    if (!new_detector) {
      spdlog::error("Failed to get detector for model: {}", model_id);
      return false;
    }
    g_detector_cache[cache_key] = std::move(new_detector);
    cache_it = g_detector_cache.find(cache_key);
  }

  jrb::infrastructure::cuda::IYoloDetector* yolo_detector = cache_it->second.get();

  if (!pipeline_has_output) {
    // Model inference only: preserve original frame as the output image.
    const size_t input_size =
        static_cast<size_t>(input_buffer.width) * input_buffer.height * input_buffer.channels;
    const size_t output_size =
        static_cast<size_t>(output_buffer.width) * output_buffer.height * output_buffer.channels;
    std::memcpy(output_buffer.data, input_buffer.data, std::min(input_size, output_size));
  }

  // Run detector on the original RGB input for accuracy; detections are stored internally.
  std::vector<unsigned char> detector_passthrough(static_cast<size_t>(input_buffer.width) *
                                                  input_buffer.height * input_buffer.channels);
  jrb::domain::interfaces::ImageBufferMut detector_output(
      detector_passthrough.data(), input_buffer.width, input_buffer.height, input_buffer.channels);
  jrb::domain::interfaces::FilterContext det_context(input_buffer.data, detector_output.data,
                                                     input_buffer.width, input_buffer.height,
                                                     input_buffer.channels);
  det_context.output = detector_output;
  yolo_detector->Apply(det_context);

  const auto& detections = yolo_detector->GetDetections();
  spdlog::info("YOLO: {} detection(s) for {}x{} image (confidence threshold {})", detections.size(),
               input_buffer.width, input_buffer.height, confidence);
  for (const auto& det : detections) {
    spdlog::info("  → {} ({:.0f}%) at [{:.0f},{:.0f} {}x{}]", det.class_name,
                 det.confidence * 100.0f, det.x, det.y, det.width, det.height);
    auto* msg = response->add_detections();
    msg->set_x(det.x);
    msg->set_y(det.y);
    msg->set_width(det.width);
    msg->set_height(det.height);
    msg->set_class_id(det.class_id);
    msg->set_class_name(det.class_name);
    msg->set_confidence(det.confidence);
  }

  return true;
}

}  // namespace jrb::application::engine::platform::cuda

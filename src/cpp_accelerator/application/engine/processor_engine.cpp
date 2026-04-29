#include "src/cpp_accelerator/application/engine/processor_engine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <utility>

#include "src/cpp_accelerator/core/version.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/application/engine/filter_descriptor.h"
#include "src/cpp_accelerator/application/pipeline/filter_pipeline.h"
#include "src/cpp_accelerator/core/logger.h"
#include "src/cpp_accelerator/core/telemetry.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"
#include "src/cpp_accelerator/adapters/compute/cpu/cpu_filter_factory.h"
#include "src/cpp_accelerator/adapters/compute/cuda/cuda_filter_factory.h"
#include "src/cpp_accelerator/adapters/compute/opencl/opencl_filter_factory.h"
#include "src/cpp_accelerator/adapters/compute/cuda/tensorrt/model_manager.h"
#include "src/cpp_accelerator/adapters/compute/cuda/tensorrt/model_registry.h"

namespace jrb::application::engine {

using jrb::domain::interfaces::GrayscaleAlgorithm;

namespace {

BlurBorderMode ProtoToBlurBorderMode(cuda_learning::BorderMode mode) {
  switch (mode) {
    case cuda_learning::BORDER_MODE_CLAMP:
      return BlurBorderMode::CLAMP;
    case cuda_learning::BORDER_MODE_WRAP:
      return BlurBorderMode::WRAP;
    default:
      return BlurBorderMode::REFLECT;
  }
}

}  // namespace

ProcessorEngine::ProcessorEngine(std::string component_name)
    : component_name_(std::move(component_name)) {
  factory_registry_.Register(std::make_unique<jrb::infrastructure::cuda::CudaFilterFactory>());
  factory_registry_.Register(std::make_unique<jrb::infrastructure::cpu::CpuFilterFactory>());
  factory_registry_.Register(
      std::make_unique<jrb::infrastructure::opencl::OpenCLFilterFactory>());
}

bool ProcessorEngine::Initialize(const cuda_learning::InitRequest& request,
                                 cuda_learning::InitResponse* response) {
  static bool logger_initialized = false;
  if (!logger_initialized) {
    jrb::core::initialize_logger();
    auto logger = spdlog::default_logger();
    if (logger) {
      logger->flush();
    }
    logger_initialized = true;
  }

  if (!response) {
    return false;
  }

  spdlog::info("Initializing CUDA context (device: {})", request.cuda_device_id());

  try {
    auto& telemetry = jrb::core::telemetry::TelemetryManager::GetInstance();
    telemetry.Initialize("cuda-image-processor-cpp", "localhost:4317", true);

    response->set_code(0);
    response->set_message("CUDA context and telemetry initialized successfully");
    spdlog::info("Initialization successful (Telemetry)");

    auto& model_manager = jrb::infrastructure::cuda::ModelManager::GetInstance();
    jrb::infrastructure::cuda::ModelRegistry registry;
    registry.RegisterModel(
        {"yolov10n", "YOLO v10 Nano", "data/models/yolov10n.onnx", "Fastest YOLO v10 model"});
    model_manager.Initialize(registry);
    spdlog::info("Model manager initialized with {} models",
                 model_manager.GetAvailableModels().size());

    // Pre-load all TRT engines at startup to avoid long latency on the first request.
    // Engine build from ONNX takes ~60-90s on x86 / ~120s on Jetson on first run;
    // subsequent starts load the cached .engine file in <1s.
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
        detector_cache_[cache_key] = std::move(detector);
        spdlog::info("[Startup] TRT engine ready for '{}' ({}s)", model_id, elapsed);
      } else {
        spdlog::error("[Startup] Failed to load TRT engine for '{}' after {}s", model_id, elapsed);
      }
    }
    spdlog::info("[Startup] All detectors loaded — ready to accept connections");
  } catch (const std::exception& e) {
    spdlog::error("Initialization failed: {}", e.what());
    response->set_code(2);
    response->set_message(std::string("Initialization failed: ") + e.what());
    return false;
  }

  return true;
}

bool ProcessorEngine::ProcessImage(const cuda_learning::ProcessImageRequest& request,
                                   cuda_learning::ProcessImageResponse* response,
                                   void* memory_pool) {
  if (!response) {
    return false;
  }
  return ApplyFilters(request, response, memory_pool);
}

bool ProcessorEngine::GetCapabilities(cuda_learning::GetCapabilitiesResponse* response,
                                      cuda_learning::AcceleratorType requested_accelerator) {
  if (!response) {
    return false;
  }

  response->set_code(0);
  response->set_message("OK");

  auto* caps = response->mutable_capabilities();
  caps->set_api_version("2.1.0");
  caps->set_library_version(kLibraryVersionStr);
  caps->set_supports_streaming(false);
  caps->set_build_date(__DATE__);
  caps->set_build_commit(kLibraryGitHashStr);

  // When a specific accelerator is requested, return only that accelerator's filters.
  // When unspecified, return all factories' filters (backward compatibility).
  auto populate_from_factory = [&](IFilterFactory* factory) {
    if (!factory) return;
    for (const auto& fd : factory->GetFilterDescriptors()) {
      auto* filter = caps->add_filters();
      filter->set_id(fd.id);
      filter->set_name(fd.name);
      filter->add_supported_accelerators(factory->GetAcceleratorType());
      for (const auto& pd : fd.parameters) {
        auto* param = filter->add_parameters();
        param->set_id(pd.id);
        param->set_name(pd.name);
        param->set_type(pd.type);
        param->set_default_value(pd.default_value);
        for (const auto& opt : pd.options) {
          param->add_options(opt.value);
        }
        auto* meta = param->mutable_metadata();
        for (const auto& [k, v] : pd.metadata) {
          (*meta)[k] = v;
        }
      }
    }
  };

  if (requested_accelerator != cuda_learning::ACCELERATOR_TYPE_UNSPECIFIED) {
    populate_from_factory(factory_registry_.GetFactory(requested_accelerator));
  } else {
    for (auto acc : factory_registry_.GetRegisteredTypes()) {
      populate_from_factory(factory_registry_.GetFactory(acc));
    }
  }

  return true;
}

GrayscaleAlgorithm ProcessorEngine::ProtoToAlgorithm(cuda_learning::GrayscaleType type) const {
  switch (type) {
    case cuda_learning::GRAYSCALE_TYPE_BT601:
      return GrayscaleAlgorithm::BT601;
    case cuda_learning::GRAYSCALE_TYPE_BT709:
      return GrayscaleAlgorithm::BT709;
    case cuda_learning::GRAYSCALE_TYPE_AVERAGE:
      return GrayscaleAlgorithm::Average;
    case cuda_learning::GRAYSCALE_TYPE_LIGHTNESS:
      return GrayscaleAlgorithm::Lightness;
    case cuda_learning::GRAYSCALE_TYPE_LUMINOSITY:
      return GrayscaleAlgorithm::Luminosity;
    default:
      return GrayscaleAlgorithm::BT601;
  }
}

bool ProcessorEngine::ApplyFilters(const cuda_learning::ProcessImageRequest& request,
                                   cuda_learning::ProcessImageResponse* response,
                                   void* memory_pool) {
  cuda_learning::AcceleratorType accelerator = request.accelerator();
  if (accelerator == cuda_learning::ACCELERATOR_TYPE_UNSPECIFIED) {
    accelerator = cuda_learning::ACCELERATOR_TYPE_CUDA;
  }

  cuda_learning::GrayscaleType grayscale_type = request.grayscale_type();
  if (grayscale_type == cuda_learning::GRAYSCALE_TYPE_UNSPECIFIED) {
    grayscale_type = cuda_learning::GRAYSCALE_TYPE_BT601;
  }

  auto& telemetry = jrb::core::telemetry::TelemetryManager::GetInstance();
  auto span =
      telemetry.CreateChildSpan(component_name_, "ProcessImage", request.trace_id(),
                                request.span_id(), static_cast<uint8_t>(request.trace_flags()));
  jrb::core::telemetry::ScopedSpan scoped_span(span);

  scoped_span.SetAttribute("image.width", static_cast<int64_t>(request.width()));
  scoped_span.SetAttribute("image.height", static_cast<int64_t>(request.height()));
  scoped_span.SetAttribute("image.channels", static_cast<int64_t>(request.channels()));
  scoped_span.SetAttribute("accelerator", cuda_learning::AcceleratorType_Name(accelerator));
  scoped_span.SetAttribute("grayscale_type", cuda_learning::GrayscaleType_Name(grayscale_type));
  scoped_span.SetAttribute("filters_count", static_cast<int64_t>(request.filters_size()));

  try {
    jrb::application::pipeline::FilterPipeline pipeline;
    jrb::infrastructure::cuda::IYoloDetector* yolo_detector = nullptr;

    IFilterFactory* factory = factory_registry_.GetFactory(accelerator);

    for (int i = 0; i < request.filters_size(); i++) {
      const auto filter = request.filters(i);
      if (filter == cuda_learning::FILTER_TYPE_NONE) {
        continue;
      }

      if (filter == cuda_learning::FILTER_TYPE_GRAYSCALE) {
        FilterCreationParams params;
        params.grayscale_algorithm = ProtoToAlgorithm(grayscale_type);
        if (factory) {
          auto f = factory->CreateFilter(jrb::domain::interfaces::FilterType::GRAYSCALE, params);
          if (f) {
            scoped_span.AddEvent("Added grayscale filter to pipeline");
            pipeline.AddFilter(std::move(f));
          } else {
            spdlog::warn("Grayscale filter not supported by accelerator {}",
                         cuda_learning::AcceleratorType_Name(accelerator));
          }
        }
      } else if (filter == cuda_learning::FILTER_TYPE_BLUR) {
        FilterCreationParams params;
        if (request.has_blur_params()) {
          const auto& bp = request.blur_params();
          params.blur_kernel_size = bp.kernel_size() > 0 ? bp.kernel_size() : 5;
          params.blur_sigma = bp.sigma() > 0.0F ? bp.sigma() : 1.0F;
          params.blur_separable = bp.separable();
          params.blur_border_mode = ProtoToBlurBorderMode(bp.border_mode());
        }
        if (factory) {
          auto f = factory->CreateFilter(jrb::domain::interfaces::FilterType::BLUR, params);
          if (f) {
            scoped_span.AddEvent("Added blur filter to pipeline");
            pipeline.AddFilter(std::move(f));
          } else {
            spdlog::warn("Blur filter not supported by accelerator {}",
                         cuda_learning::AcceleratorType_Name(accelerator));
          }
        }
      } else if (filter == cuda_learning::FILTER_TYPE_MODEL_INFERENCE) {
        std::string model_id = "yolov10n";
        float confidence = 0.5f;

        if (request.has_model_params()) {
          if (!request.model_params().model_id().empty()) {
            model_id = request.model_params().model_id();
          }
          confidence = request.model_params().confidence_threshold() > 0
                           ? request.model_params().confidence_threshold()
                           : 0.5f;
        }

        std::string cache_key = model_id + "@" + std::to_string(confidence);
        auto cache_it = detector_cache_.find(cache_key);
        if (cache_it == detector_cache_.end()) {
          auto new_detector = jrb::infrastructure::cuda::ModelManager::GetInstance().GetDetector(
              model_id, confidence);
          if (!new_detector) {
            spdlog::error("Failed to get detector for model: {}", model_id);
          } else {
            detector_cache_[cache_key] = std::move(new_detector);
            cache_it = detector_cache_.find(cache_key);
          }
        }
        if (cache_it != detector_cache_.end()) {
          yolo_detector = cache_it->second.get();
          scoped_span.AddEvent("Using cached YOLO model inference filter");
        }
      } else {
        spdlog::warn("Unsupported filter type: {}", filter);
      }
    }

    if (pipeline.GetFilterCount() == 0 && yolo_detector == nullptr) {
      spdlog::error("No valid filters to apply");
      response->set_code(5);
      response->set_message("No valid filters to apply");
      return false;
    }

    jrb::domain::interfaces::ImageBuffer input_buffer(
        reinterpret_cast<const unsigned char*>(request.image_data().data()), request.width(),
        request.height(), request.channels());

    int output_channels = request.channels();
    for (int i = 0; i < request.filters_size(); i++) {
      if (request.filters(i) == cuda_learning::FILTER_TYPE_GRAYSCALE) {
        output_channels = 1;
        break;
      }
    }

    std::vector<unsigned char> output_data(request.width() * request.height() * output_channels);
    jrb::domain::interfaces::ImageBufferMut output_buffer(output_data.data(), request.width(),
                                                          request.height(), output_channels);

    if (pipeline.GetFilterCount() > 0) {
      bool success = pipeline.Apply(input_buffer, output_buffer, memory_pool);
      if (!success) {
        spdlog::error("Filter pipeline processing failed");
        scoped_span.RecordError("Filter pipeline processing failed");
        response->set_code(7);
        response->set_message("Filter pipeline processing failed");
        return false;
      }
    } else if (yolo_detector != nullptr) {
      // Model inference only: keep original frame visible while detections are produced.
      const size_t input_size =
          static_cast<size_t>(input_buffer.width) * input_buffer.height * input_buffer.channels;
      const size_t output_size =
          static_cast<size_t>(output_buffer.width) * output_buffer.height * output_buffer.channels;
      const size_t copy_size = std::min(input_size, output_size);
      std::memcpy(output_buffer.data, input_buffer.data, copy_size);
    }

    if (yolo_detector != nullptr) {
      // Keep detector input on original RGB for accuracy, but never let YOLO's passthrough write
      // override the already-processed pipeline output.
      std::vector<unsigned char> detector_passthrough(static_cast<size_t>(input_buffer.width) *
                                                      input_buffer.height * input_buffer.channels);
      jrb::domain::interfaces::ImageBufferMut detector_output(
          detector_passthrough.data(), input_buffer.width, input_buffer.height,
          input_buffer.channels);
      jrb::domain::interfaces::FilterContext det_context(input_buffer.data, detector_output.data,
                                                         input_buffer.width, input_buffer.height,
                                                         input_buffer.channels);
      det_context.output = detector_output;
      yolo_detector->Apply(det_context);
    }

    response->set_code(0);
    response->set_message("Image processed successfully");
    response->set_image_data(output_data.data(), output_data.size());
    response->set_width(output_buffer.width);
    response->set_height(output_buffer.height);
    response->set_channels(output_buffer.channels);

    // Populate detections if a detector was used
    if (yolo_detector != nullptr) {
      const auto& detections = yolo_detector->GetDetections();
      spdlog::info(
          "YOLO: {} detection(s) for {}x{} image (confidence threshold {})", detections.size(),
          request.width(), request.height(),
          request.has_model_params() ? request.model_params().confidence_threshold() : 0.5f);
      for (const auto& det : detections) {
        spdlog::info("  → {} ({:.0f}%) at [{:.0f},{:.0f} {}x{}]", det.class_name,
                     det.confidence * 100.0f, det.x, det.y, det.width, det.height);
        auto* detection_msg = response->add_detections();
        detection_msg->set_x(det.x);
        detection_msg->set_y(det.y);
        detection_msg->set_width(det.width);
        detection_msg->set_height(det.height);
        detection_msg->set_class_id(det.class_id);
        detection_msg->set_class_name(det.class_name);
        detection_msg->set_confidence(det.confidence);
      }
      scoped_span.SetAttribute("detections.count", static_cast<int64_t>(detections.size()));
    }

    scoped_span.SetAttribute("result.width", static_cast<int64_t>(output_buffer.width));
    scoped_span.SetAttribute("result.height", static_cast<int64_t>(output_buffer.height));
    scoped_span.SetAttribute("result.channels", static_cast<int64_t>(output_buffer.channels));

    return true;
  } catch (const std::exception& e) {
    spdlog::error("Exception during image processing: {}", e.what());
    scoped_span.RecordError(std::string("Exception: ") + e.what());
    response->set_code(6);
    response->set_message(std::string("Exception: ") + e.what());
    return false;
  }
}

}  // namespace jrb::application::engine

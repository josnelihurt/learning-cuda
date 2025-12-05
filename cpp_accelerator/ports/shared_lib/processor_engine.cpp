#include "cpp_accelerator/ports/shared_lib/processor_engine.h"

#include <cstring>
#include <string>
#include <utility>

#include "cpp_accelerator/ports/shared_lib/library_version.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "cpp_accelerator/application/pipeline/filter_pipeline.h"
#include "cpp_accelerator/core/logger.h"
#include "cpp_accelerator/core/telemetry.h"
#include "cpp_accelerator/domain/interfaces/image_buffer.h"
#include "cpp_accelerator/infrastructure/cpu/blur_filter.h"
#include "cpp_accelerator/infrastructure/cpu/grayscale_filter.h"
#include "cpp_accelerator/infrastructure/cuda/blur_processor.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_filter.h"

namespace jrb::ports::shared_lib {

using jrb::domain::interfaces::GrayscaleAlgorithm;

namespace {

jrb::infrastructure::cpu::BorderMode ProtoToBorderMode(cuda_learning::BorderMode mode) {
  switch (mode) {
    case cuda_learning::BORDER_MODE_CLAMP:
      return jrb::infrastructure::cpu::BorderMode::CLAMP;
    case cuda_learning::BORDER_MODE_REFLECT:
      return jrb::infrastructure::cpu::BorderMode::REFLECT;
    case cuda_learning::BORDER_MODE_WRAP:
      return jrb::infrastructure::cpu::BorderMode::WRAP;
    default:
      return jrb::infrastructure::cpu::BorderMode::REFLECT;
  }
}

jrb::infrastructure::cuda::BorderMode ProtoToCudaBorderMode(cuda_learning::BorderMode mode) {
  switch (mode) {
    case cuda_learning::BORDER_MODE_CLAMP:
      return jrb::infrastructure::cuda::BorderMode::CLAMP;
    case cuda_learning::BORDER_MODE_REFLECT:
      return jrb::infrastructure::cuda::BorderMode::REFLECT;
    case cuda_learning::BORDER_MODE_WRAP:
      return jrb::infrastructure::cuda::BorderMode::WRAP;
    default:
      return jrb::infrastructure::cuda::BorderMode::REFLECT;
  }
}

}  // namespace

ProcessorEngine::ProcessorEngine(std::string component_name)
    : component_name_(std::move(component_name)) {}

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
  } catch (const std::exception& e) {
    spdlog::error("Initialization failed: {}", e.what());
    response->set_code(2);
    response->set_message(std::string("Initialization failed: ") + e.what());
    return false;
  }

  return true;
}

bool ProcessorEngine::ProcessImage(const cuda_learning::ProcessImageRequest& request,
                                   cuda_learning::ProcessImageResponse* response) {
  if (!response) {
    return false;
  }
  return ApplyFilters(request, response);
}

bool ProcessorEngine::GetCapabilities(cuda_learning::GetCapabilitiesResponse* response) {
  if (!response) {
    return false;
  }

  response->set_code(0);
  response->set_message("OK");

  auto* caps = response->mutable_capabilities();
  caps->set_api_version(PROCESSOR_API_VERSION);
  caps->set_library_version(LIBRARY_VERSION_STR);
  caps->set_supports_streaming(false);
  caps->set_build_date(__DATE__);
#ifdef BUILD_COMMIT
  caps->set_build_commit(BUILD_COMMIT);
#else
  caps->set_build_commit("unknown");
#endif

  auto* grayscale_filter = caps->add_filters();
  grayscale_filter->set_id("grayscale");
  grayscale_filter->set_name("Grayscale");
  grayscale_filter->add_supported_accelerators(cuda_learning::ACCELERATOR_TYPE_CUDA);
  grayscale_filter->add_supported_accelerators(cuda_learning::ACCELERATOR_TYPE_CPU);

  auto* algorithm_param = grayscale_filter->add_parameters();
  algorithm_param->set_id("algorithm");
  algorithm_param->set_name("Algorithm");
  algorithm_param->set_type("select");
  algorithm_param->add_options("bt601");
  algorithm_param->add_options("bt709");
  algorithm_param->add_options("average");
  algorithm_param->add_options("lightness");
  algorithm_param->add_options("luminosity");
  algorithm_param->set_default_value("bt601");

  auto* blur_filter = caps->add_filters();
  blur_filter->set_id("blur");
  blur_filter->set_name("Gaussian Blur");
  blur_filter->add_supported_accelerators(cuda_learning::ACCELERATOR_TYPE_CUDA);
  blur_filter->add_supported_accelerators(cuda_learning::ACCELERATOR_TYPE_CPU);

  auto* kernel_size_param = blur_filter->add_parameters();
  kernel_size_param->set_id("kernel_size");
  kernel_size_param->set_name("Kernel Size");
  kernel_size_param->set_type("range");
  kernel_size_param->set_default_value("5");

  auto* sigma_param = blur_filter->add_parameters();
  sigma_param->set_id("sigma");
  sigma_param->set_name("Sigma");
  sigma_param->set_type("number");
  sigma_param->set_default_value("1.0");

  auto* border_mode_param = blur_filter->add_parameters();
  border_mode_param->set_id("border_mode");
  border_mode_param->set_name("Border Mode");
  border_mode_param->set_type("select");
  border_mode_param->add_options("CLAMP");
  border_mode_param->add_options("REFLECT");
  border_mode_param->add_options("WRAP");
  border_mode_param->set_default_value("REFLECT");

  auto* separable_param = blur_filter->add_parameters();
  separable_param->set_id("separable");
  separable_param->set_name("Separable");
  separable_param->set_type("checkbox");
  separable_param->set_default_value("true");

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
                                   cuda_learning::ProcessImageResponse* response) {
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

    for (int i = 0; i < request.filters_size(); i++) {
      const auto filter = request.filters(i);
      if (filter == cuda_learning::FILTER_TYPE_NONE) {
        continue;
      }

      if (filter == cuda_learning::FILTER_TYPE_GRAYSCALE) {
        GrayscaleAlgorithm algorithm = ProtoToAlgorithm(grayscale_type);
        if (accelerator == cuda_learning::ACCELERATOR_TYPE_CUDA) {
          pipeline.AddFilter(
              std::make_unique<jrb::infrastructure::cuda::GrayscaleFilter>(algorithm));
          scoped_span.AddEvent("Added CUDA grayscale filter to pipeline");
        } else {
          pipeline.AddFilter(
              std::make_unique<jrb::infrastructure::cpu::GrayscaleFilter>(algorithm));
          scoped_span.AddEvent("Added CPU grayscale filter to pipeline");
        }
      } else if (filter == cuda_learning::FILTER_TYPE_BLUR) {
        int kernel_size = 5;
        float sigma = 1.0F;
        bool separable = true;

        if (request.has_blur_params()) {
          const auto& blur_params = request.blur_params();
          kernel_size = blur_params.kernel_size() > 0 ? blur_params.kernel_size() : 5;
          sigma = blur_params.sigma() > 0.0F ? blur_params.sigma() : 1.0F;
          separable = blur_params.separable();
        }

        if (accelerator == cuda_learning::ACCELERATOR_TYPE_CUDA) {
          auto border_mode = request.has_blur_params()
                                 ? ProtoToCudaBorderMode(request.blur_params().border_mode())
                                 : jrb::infrastructure::cuda::BorderMode::REFLECT;

          pipeline.AddFilter(std::make_unique<jrb::infrastructure::cuda::CudaGaussianBlurFilter>(
              kernel_size, sigma, border_mode, separable));
          scoped_span.AddEvent("Added CUDA blur filter to pipeline");
        } else {
          auto border_mode = request.has_blur_params()
                                 ? ProtoToBorderMode(request.blur_params().border_mode())
                                 : jrb::infrastructure::cpu::BorderMode::REFLECT;
          pipeline.AddFilter(std::make_unique<jrb::infrastructure::cpu::GaussianBlurFilter>(
              kernel_size, sigma, border_mode, separable));
          scoped_span.AddEvent("Added CPU blur filter to pipeline");
        }
      } else {
        spdlog::warn("Unsupported filter type: {}", filter);
      }
    }

    if (pipeline.GetFilterCount() == 0) {
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

    bool success = pipeline.Apply(input_buffer, output_buffer);
    if (!success) {
      spdlog::error("Filter pipeline processing failed");
      scoped_span.RecordError("Filter pipeline processing failed");
      response->set_code(7);
      response->set_message("Filter pipeline processing failed");
      return false;
    }

    response->set_code(0);
    response->set_message("Image processed successfully");
    response->set_image_data(output_data.data(), output_data.size());
    response->set_width(output_buffer.width);
    response->set_height(output_buffer.height);
    response->set_channels(output_buffer.channels);

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

}  // namespace jrb::ports::shared_lib

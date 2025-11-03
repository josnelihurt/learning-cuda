#include <spdlog/spdlog.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "common.pb.h"
#include "cpp_accelerator/application/pipeline/filter_pipeline.h"
#include "cpp_accelerator/core/logger.h"
#include "cpp_accelerator/core/telemetry.h"
#include "cpp_accelerator/domain/interfaces/grayscale_algorithm.h"
#include "cpp_accelerator/infrastructure/cpu/blur_filter.h"
#include "cpp_accelerator/infrastructure/cpu/grayscale_filter.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_filter.h"
#include "image_buffer_adapter.h"
#include "image_processor_service.pb.h"
#include "processor_api.h"

namespace {

using jrb::domain::interfaces::GrayscaleAlgorithm;

// Helper to allocate response buffer (must be freed by caller with FreeResponse)
uint8_t* allocate_response(const std::string& serialized, int* out_len) {
  *out_len = static_cast<int>(serialized.size());
  uint8_t* buffer = new uint8_t[*out_len];
  std::memcpy(buffer, serialized.data(), *out_len);
  return buffer;
}

GrayscaleAlgorithm proto_to_algorithm(cuda_learning::GrayscaleType type) {
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

// Convert protobuf BorderMode to C++ BorderMode
jrb::infrastructure::cpu::BorderMode proto_to_border_mode(cuda_learning::BorderMode mode) {
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

}  // anonymous namespace

extern "C" {

processor_version_t processor_api_version(void) {
  processor_version_t version;
  version.major = (PROCESSOR_API_VERNUM >> 16) & 0xFF;
  version.minor = (PROCESSOR_API_VERNUM >> 8) & 0xFF;
  version.patch = PROCESSOR_API_VERNUM & 0xFF;
  return version;
}

bool processor_init(const uint8_t* request, int request_len, uint8_t** response,
                    int* response_len) {
  static bool logger_initialized = false;
  if (!logger_initialized) {
    // Initialize logger BEFORE any other spdlog calls
    jrb::core::initialize_logger();

    // Verify logger is active and flush to ensure file is created
    auto logger = spdlog::default_logger();
    if (logger) {
      logger->flush();
    }

    logger_initialized = true;
  }

  cuda_learning::InitRequest init_req;
  cuda_learning::InitResponse init_resp;

  // Parse request
  if (!init_req.ParseFromArray(request, request_len)) {
    spdlog::error("Failed to parse InitRequest");
    init_resp.set_code(1);
    init_resp.set_message("Failed to parse InitRequest");
    *response = allocate_response(init_resp.SerializeAsString(), response_len);
    return false;
  }

  spdlog::info("Initializing CUDA context (device: {})", init_req.cuda_device_id());

  try {
    auto& telemetry = jrb::core::telemetry::TelemetryManager::GetInstance();
    telemetry.Initialize("cuda-image-processor-cpp", "localhost:4317", true);

    init_resp.set_code(0);
    init_resp.set_message("CUDA context and telemetry initialized successfully");
    spdlog::info("Initialization successful (Telemetry)");
  } catch (const std::exception& e) {
    spdlog::error("Initialization failed: {}", e.what());
    init_resp.set_code(2);
    init_resp.set_message(std::string("Initialization failed: ") + e.what());
    *response = allocate_response(init_resp.SerializeAsString(), response_len);
    return false;
  }

  *response = allocate_response(init_resp.SerializeAsString(), response_len);
  return true;
}

void processor_cleanup() {
  spdlog::info("Cleaning up CUDA context and telemetry");

  auto& telemetry = jrb::core::telemetry::TelemetryManager::GetInstance();
  telemetry.Shutdown();
}

bool processor_process_image(const uint8_t* request, int request_len, uint8_t** response,
                             int* response_len) {
  cuda_learning::ProcessImageRequest proc_req;
  cuda_learning::ProcessImageResponse proc_resp;

  // Parse request
  if (!proc_req.ParseFromArray(request, request_len)) {
    spdlog::error("Failed to parse ProcessImageRequest");
    proc_resp.set_code(1);
    proc_resp.set_message("Failed to parse ProcessImageRequest");
    *response = allocate_response(proc_resp.SerializeAsString(), response_len);
    return false;
  }

  // Create span with parent context from Go
  auto& telemetry = jrb::core::telemetry::TelemetryManager::GetInstance();
  auto span =
      telemetry.CreateChildSpan("cgo-api", "ProcessImage", proc_req.trace_id(), proc_req.span_id(),
                                static_cast<uint8_t>(proc_req.trace_flags()));
  jrb::core::telemetry::ScopedSpan scoped_span(span);

  // Get accelerator type (default to GPU if not specified)
  cuda_learning::AcceleratorType accelerator = proc_req.accelerator();
  if (accelerator == cuda_learning::ACCELERATOR_TYPE_UNSPECIFIED) {
    accelerator = cuda_learning::ACCELERATOR_TYPE_CUDA;
  }

  // Get grayscale algorithm type (default to BT601 if not specified)
  cuda_learning::GrayscaleType grayscale_type = proc_req.grayscale_type();
  if (grayscale_type == cuda_learning::GRAYSCALE_TYPE_UNSPECIFIED) {
    grayscale_type = cuda_learning::GRAYSCALE_TYPE_BT601;
  }

  spdlog::debug("Processing image: {}x{}, {} channels, accelerator: {}, grayscale_type: {}",
                proc_req.width(), proc_req.height(), proc_req.channels(),
                cuda_learning::AcceleratorType_Name(accelerator),
                cuda_learning::GrayscaleType_Name(grayscale_type));

  scoped_span.SetAttribute("image.width", static_cast<int64_t>(proc_req.width()));
  scoped_span.SetAttribute("image.height", static_cast<int64_t>(proc_req.height()));
  scoped_span.SetAttribute("image.channels", static_cast<int64_t>(proc_req.channels()));
  scoped_span.SetAttribute("accelerator", cuda_learning::AcceleratorType_Name(accelerator));
  scoped_span.SetAttribute("grayscale_type", cuda_learning::GrayscaleType_Name(grayscale_type));
  scoped_span.SetAttribute("filters_count", static_cast<int64_t>(proc_req.filters_size()));

  try {
    jrb::application::pipeline::FilterPipeline pipeline;
    using jrb::domain::interfaces::ImageBuffer;
    using jrb::domain::interfaces::ImageBufferMut;

    for (int i = 0; i < proc_req.filters_size(); i++) {
      cuda_learning::FilterType filter = proc_req.filters(i);

      if (filter == cuda_learning::FILTER_TYPE_NONE) {
        continue;
      }

      if (filter == cuda_learning::FILTER_TYPE_GRAYSCALE) {
        GrayscaleAlgorithm algorithm = proto_to_algorithm(grayscale_type);
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
        jrb::infrastructure::cpu::BorderMode border_mode =
            jrb::infrastructure::cpu::BorderMode::REFLECT;
        bool separable = true;

        if (proc_req.has_blur_params()) {
          const auto& blur_params = proc_req.blur_params();
          kernel_size = blur_params.kernel_size() > 0 ? blur_params.kernel_size() : 5;
          sigma = blur_params.sigma() > 0.0F ? blur_params.sigma() : 1.0F;
          border_mode = proto_to_border_mode(blur_params.border_mode());
          separable = blur_params.separable();
        }

        pipeline.AddFilter(std::make_unique<jrb::infrastructure::cpu::GaussianBlurFilter>(
            kernel_size, sigma, border_mode, separable));
        scoped_span.AddEvent("Added blur filter to pipeline");
      } else {
        spdlog::warn("Unsupported filter type: {}", filter);
      }
    }

    if (pipeline.GetFilterCount() == 0) {
      spdlog::error("No valid filters to apply");
      proc_resp.set_code(5);
      proc_resp.set_message("No valid filters to apply");
      *response = allocate_response(proc_resp.SerializeAsString(), response_len);
      return false;
    }

    ImageBuffer input_buffer(reinterpret_cast<const unsigned char*>(proc_req.image_data().data()),
                             proc_req.width(), proc_req.height(), proc_req.channels());

    int output_channels = proc_req.channels();
    for (int i = 0; i < proc_req.filters_size(); i++) {
      if (proc_req.filters(i) == cuda_learning::FILTER_TYPE_GRAYSCALE) {
        output_channels = 1;
        break;
      }
    }

    std::vector<unsigned char> output_data(proc_req.width() * proc_req.height() * output_channels);
    ImageBufferMut output_buffer(output_data.data(), proc_req.width(), proc_req.height(),
                                 output_channels);

    bool success = pipeline.Apply(input_buffer, output_buffer);

    if (!success) {
      spdlog::error("Filter pipeline processing failed");
      scoped_span.RecordError("Filter pipeline processing failed");
      proc_resp.set_code(7);
      proc_resp.set_message("Filter pipeline processing failed");
      *response = allocate_response(proc_resp.SerializeAsString(), response_len);
      return false;
    }

    proc_resp.set_code(0);
    proc_resp.set_message("Image processed successfully");
    proc_resp.set_image_data(output_data.data(), output_data.size());
    proc_resp.set_width(output_buffer.width);
    proc_resp.set_height(output_buffer.height);
    proc_resp.set_channels(output_buffer.channels);

    spdlog::debug("Image processing completed successfully");

    scoped_span.SetAttribute("result.width", static_cast<int64_t>(output_buffer.width));
    scoped_span.SetAttribute("result.height", static_cast<int64_t>(output_buffer.height));
    scoped_span.SetAttribute("result.channels", static_cast<int64_t>(output_buffer.channels));

  } catch (const std::exception& e) {
    spdlog::error("Exception during image processing: {}", e.what());
    scoped_span.RecordError(std::string("Exception: ") + e.what());
    proc_resp.set_code(6);
    proc_resp.set_message(std::string("Exception: ") + e.what());
    *response = allocate_response(proc_resp.SerializeAsString(), response_len);
    return false;
  }

  *response = allocate_response(proc_resp.SerializeAsString(), response_len);
  return true;
}

bool processor_get_capabilities(const uint8_t* request, int request_len, uint8_t** response,
                                int* response_len) {
  (void)request;
  (void)request_len;

  cuda_learning::GetCapabilitiesResponse resp;
  resp.set_code(0);
  resp.set_message("OK");

  auto* caps = resp.mutable_capabilities();
  caps->set_api_version(PROCESSOR_API_VERSION);
  char version_buf[64];
  std::string library_version;
  if (processor_get_library_version(version_buf, sizeof(version_buf))) {
    library_version = version_buf;
  }
  if (library_version.empty()) {
    library_version = PROCESSOR_API_VERSION;
  }
  caps->set_library_version(library_version);
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

  *response = allocate_response(resp.SerializeAsString(), response_len);
  return true;
}

bool processor_get_library_version(char* version_buf, int buf_len) {
  spdlog::info("GetLibraryVersion request received - fetching library version");

  if (!version_buf || buf_len < 1) {
    spdlog::error("Invalid buffer for GetLibraryVersion");
    return false;
  }

  version_buf[0] = '\0';

  std::string version;
  bool found = false;

  static const char* version_file_paths[] = {"cpp_accelerator/VERSION",
                                             "../cpp_accelerator/VERSION",
                                             "../../cpp_accelerator/VERSION", "./VERSION", nullptr};

  for (int i = 0; version_file_paths[i] != nullptr && !found; ++i) {
    std::ifstream file(version_file_paths[i]);
    if (file.is_open()) {
      std::getline(file, version);
      file.close();
      if (!version.empty()) {
        found = true;
        spdlog::info("Library version loaded from file: {} = {}", version_file_paths[i], version);
      }
    }
  }

  if (!found) {
    version = PROCESSOR_API_VERSION;
    spdlog::warn("VERSION file not found, using API version as fallback: {}", version);
  }

  size_t copy_len = version.size();
  if (copy_len >= static_cast<size_t>(buf_len)) {
    copy_len = static_cast<size_t>(buf_len) - 1;
  }

  std::strncpy(version_buf, version.c_str(), copy_len);
  version_buf[copy_len] = '\0';

  spdlog::info("GetLibraryVersion completed - returning version: {}", version_buf);
  return true;
}

void processor_free_response(uint8_t* ptr) {
  delete[] ptr;
}

}  // extern "C"

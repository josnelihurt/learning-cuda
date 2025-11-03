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
#include "cpp_accelerator/infrastructure/cpu/blur_filter.h"
#include "cpp_accelerator/infrastructure/cpu/grayscale_filter.h"
// TODO(migration): Remove these includes after creating CudaGrayscaleFilter and migrating to
// FilterPipeline Steps: See TODO at line 24
#include "cpp_accelerator/infrastructure/cpu/grayscale_processor.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_processor.h"
#include "image_buffer_adapter.h"
#include "image_processor_service.pb.h"
#include "processor_api.h"

namespace {

// TODO(migration): Remove these singleton processor instances after implementing
// CudaGrayscaleFilter. Migration plan:
// 1. Create CudaGrayscaleFilter in cpp_accelerator/infrastructure/cuda/grayscale_filter.h
// 2. Update processor_init() to remove GrayscaleProcessor initialization
// 3. Update processor_process_image() to always use FilterPipeline (remove else branch at line 286)
// 4. Update processor_cleanup() to remove processor cleanup
// 5. Remove includes for grayscale_processor.h files
// 6. Delete this TODO and the singleton instances below
std::unique_ptr<jrb::infrastructure::cuda::GrayscaleProcessor> g_cuda_grayscale_processor;
std::unique_ptr<jrb::infrastructure::cpu::CpuGrayscaleProcessor> g_cpu_grayscale_processor;

// Helper to allocate response buffer (must be freed by caller with FreeResponse)
uint8_t* allocate_response(const std::string& serialized, int* out_len) {
  *out_len = static_cast<int>(serialized.size());
  uint8_t* buffer = new uint8_t[*out_len];
  std::memcpy(buffer, serialized.data(), *out_len);
  return buffer;
}

// TODO(migration): After creating CudaGrayscaleFilter, unify these conversion functions.
// Both CPU and CUDA filters can share the same GrayscaleAlgorithm enum (move to domain/interfaces).
// Then create a single proto_to_algorithm() function that returns the unified enum.
// Convert protobuf GrayscaleType to CUDA algorithm
jrb::infrastructure::cuda::GrayscaleAlgorithm proto_to_cuda_algorithm(
    cuda_learning::GrayscaleType type) {
  switch (type) {
    case cuda_learning::GRAYSCALE_TYPE_BT601:
      return jrb::infrastructure::cuda::GrayscaleAlgorithm::BT601;
    case cuda_learning::GRAYSCALE_TYPE_BT709:
      return jrb::infrastructure::cuda::GrayscaleAlgorithm::BT709;
    case cuda_learning::GRAYSCALE_TYPE_AVERAGE:
      return jrb::infrastructure::cuda::GrayscaleAlgorithm::Average;
    case cuda_learning::GRAYSCALE_TYPE_LIGHTNESS:
      return jrb::infrastructure::cuda::GrayscaleAlgorithm::Lightness;
    case cuda_learning::GRAYSCALE_TYPE_LUMINOSITY:
      return jrb::infrastructure::cuda::GrayscaleAlgorithm::Luminosity;
    default:
      return jrb::infrastructure::cuda::GrayscaleAlgorithm::BT601;
  }
}

// Convert protobuf GrayscaleType to CPU algorithm
jrb::infrastructure::cpu::GrayscaleAlgorithm proto_to_cpu_algorithm(
    cuda_learning::GrayscaleType type) {
  switch (type) {
    case cuda_learning::GRAYSCALE_TYPE_BT601:
      return jrb::infrastructure::cpu::GrayscaleAlgorithm::BT601;
    case cuda_learning::GRAYSCALE_TYPE_BT709:
      return jrb::infrastructure::cpu::GrayscaleAlgorithm::BT709;
    case cuda_learning::GRAYSCALE_TYPE_AVERAGE:
      return jrb::infrastructure::cpu::GrayscaleAlgorithm::Average;
    case cuda_learning::GRAYSCALE_TYPE_LIGHTNESS:
      return jrb::infrastructure::cpu::GrayscaleAlgorithm::Lightness;
    case cuda_learning::GRAYSCALE_TYPE_LUMINOSITY:
      return jrb::infrastructure::cpu::GrayscaleAlgorithm::Luminosity;
    default:
      return jrb::infrastructure::cpu::GrayscaleAlgorithm::BT601;
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
    // Initialize OpenTelemetry
    auto& telemetry = jrb::core::telemetry::TelemetryManager::GetInstance();
    telemetry.Initialize("cuda-image-processor-cpp", "localhost:4317", true);

    // TODO(migration): Remove GrayscaleProcessor initialization. After implementing
    // CudaGrayscaleFilter, this initialization should be removed. FilterPipeline will create
    // filters on-demand. See migration plan at line 24
    g_cuda_grayscale_processor = std::make_unique<jrb::infrastructure::cuda::GrayscaleProcessor>();
    g_cpu_grayscale_processor = std::make_unique<jrb::infrastructure::cpu::CpuGrayscaleProcessor>();

    init_resp.set_code(0);
    init_resp.set_message("CUDA context, CPU processors, and telemetry initialized successfully");
    spdlog::info("Initialization successful (CUDA + CPU + Telemetry)");
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
  spdlog::info("Cleaning up CUDA context, CPU processors, and telemetry");
  // TODO(migration): Remove GrayscaleProcessor cleanup after removing singletons.
  // See migration plan at line 24
  g_cuda_grayscale_processor.reset();
  g_cpu_grayscale_processor.reset();

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

  // TODO(migration): Remove this check after removing GrayscaleProcessor singletons.
  // FilterPipeline creates filters on-demand, so no initialization check needed.
  // See migration plan at line 24
  if (!g_cuda_grayscale_processor || !g_cpu_grayscale_processor) {
    spdlog::error("Processors not initialized. Call CudaInit first");
    proc_resp.set_code(3);
    proc_resp.set_message("Processors not initialized. Call CudaInit first");
    *response = allocate_response(proc_resp.SerializeAsString(), response_len);
    return false;
  }

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
    bool has_blur_filter = false;
    for (int i = 0; i < proc_req.filters_size(); i++) {
      if (proc_req.filters(i) == cuda_learning::FILTER_TYPE_BLUR) {
        has_blur_filter = true;
        break;
      }
    }

    jrb::ports::cgo::ImageBufferSource source(
        reinterpret_cast<const uint8_t*>(proc_req.image_data().data()), proc_req.width(),
        proc_req.height(), proc_req.channels());

    jrb::ports::cgo::ImageBufferSink sink;
    bool success = false;

    // TODO(migration): Remove the conditional check. Always use FilterPipeline after migration.
    // Steps:
    // 1. Remove "if (has_blur_filter || proc_req.filters_size() > 1)" condition
    // 2. Remove the entire else branch (lines 286-319)
    // 3. Always create FilterPipeline and add filters based on accelerator type
    // 4. For CUDA accelerator, use CudaGrayscaleFilter instead of CpuGrayscaleFilter
    // 5. Update proto_to_cuda_algorithm and proto_to_cpu_algorithm to proto_to_algorithm (unified)
    if (has_blur_filter || proc_req.filters_size() > 1) {
      scoped_span.AddEvent("Using FilterPipeline for multi-filter processing");
      jrb::application::pipeline::FilterPipeline pipeline;
      using jrb::domain::interfaces::ImageBuffer;
      using jrb::domain::interfaces::ImageBufferMut;

      for (int i = 0; i < proc_req.filters_size(); i++) {
        cuda_learning::FilterType filter = proc_req.filters(i);

        if (filter == cuda_learning::FILTER_TYPE_NONE) {
          continue;
        }

        // TODO(migration): Use CudaGrayscaleFilter when accelerator == ACCELERATOR_TYPE_CUDA
        // Currently only CPU filter exists. After creating CudaGrayscaleFilter, add:
        // if (accelerator == cuda_learning::ACCELERATOR_TYPE_CUDA) {
        //   pipeline.AddFilter(std::make_unique<jrb::infrastructure::cuda::GrayscaleFilter>(
        //       proto_to_algorithm(grayscale_type)));
        // } else {
        //   pipeline.AddFilter(std::make_unique<jrb::infrastructure::cpu::GrayscaleFilter>(
        //       proto_to_algorithm(grayscale_type)));
        // }
        if (filter == cuda_learning::FILTER_TYPE_GRAYSCALE) {
          pipeline.AddFilter(std::make_unique<jrb::infrastructure::cpu::GrayscaleFilter>(
              proto_to_cpu_algorithm(grayscale_type)));
          scoped_span.AddEvent("Added grayscale filter to pipeline");
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
        }
      }

      ImageBuffer input_buffer(source.data(), source.width(), source.height(), source.channels());
      std::vector<unsigned char> output_data(source.width() * source.height());
      ImageBufferMut output_buffer(output_data.data(), source.width(), source.height(), 1);

      success = pipeline.Apply(input_buffer, output_buffer);

      if (!success) {
        spdlog::error("Filter pipeline processing failed");
        scoped_span.RecordError("Filter pipeline processing failed");
        proc_resp.set_code(7);
        proc_resp.set_message("Filter pipeline processing failed");
        *response = allocate_response(proc_resp.SerializeAsString(), response_len);
        return false;
      }

      sink.write("", output_data.data(), source.width(), source.height(), 1);
    } else {
      // TODO(migration): DELETE THIS ENTIRE ELSE BRANCH (lines 286-319).
      // After implementing CudaGrayscaleFilter, remove this branch and always use FilterPipeline.
      // This legacy code path uses GrayscaleProcessor directly and should be eliminated.
      // Migration steps:
      // 1. Remove the conditional check "if (has_blur_filter || proc_req.filters_size() > 1)"
      // 2. Always create FilterPipeline
      // 3. Add filters based on accelerator type (CudaGrayscaleFilter or CpuGrayscaleFilter)
      // See migration plan at line 24
      for (int i = 0; i < proc_req.filters_size(); i++) {
        cuda_learning::FilterType filter = proc_req.filters(i);

        if (filter == cuda_learning::FILTER_TYPE_NONE) {
          continue;
        }

        if (filter == cuda_learning::FILTER_TYPE_GRAYSCALE) {
          if (accelerator == cuda_learning::ACCELERATOR_TYPE_CUDA) {
            scoped_span.AddEvent("Starting CUDA grayscale processing");
            g_cuda_grayscale_processor->set_algorithm(proto_to_cuda_algorithm(grayscale_type));
            success = g_cuda_grayscale_processor->process(source, sink, "");
            scoped_span.AddEvent("CUDA grayscale processing completed");
          } else {
            scoped_span.AddEvent("Starting CPU grayscale processing");
            g_cpu_grayscale_processor->set_algorithm(proto_to_cpu_algorithm(grayscale_type));
            success = g_cpu_grayscale_processor->process(source, sink, "");
            scoped_span.AddEvent("CPU grayscale processing completed");
          }

          if (!success) {
            spdlog::error("Grayscale filter processing failed");
            scoped_span.RecordError("Grayscale filter processing failed");
            proc_resp.set_code(5);
            proc_resp.set_message("Grayscale filter processing failed");
            *response = allocate_response(proc_resp.SerializeAsString(), response_len);
            return false;
          }
        } else {
          spdlog::warn("Unsupported filter type: {}", filter);
        }
      }
    }

    // Build successful response with final output
    proc_resp.set_code(0);
    proc_resp.set_message("Image processed successfully");
    proc_resp.set_image_data(sink.get_data().data(), sink.get_data().size());
    proc_resp.set_width(sink.width());
    proc_resp.set_height(sink.height());
    proc_resp.set_channels(sink.channels());

    spdlog::debug("Image processing completed successfully");

    scoped_span.SetAttribute("result.width", static_cast<int64_t>(sink.width()));
    scoped_span.SetAttribute("result.height", static_cast<int64_t>(sink.height()));
    scoped_span.SetAttribute("result.channels", static_cast<int64_t>(sink.channels()));

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

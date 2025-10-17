#include <spdlog/spdlog.h>

#include <cstring>
#include <memory>

#include "common.pb.h"
#include "cpp_accelerator/core/telemetry.h"
#include "cpp_accelerator/infrastructure/cpu/grayscale_processor.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_processor.h"
#include "image_buffer_adapter.h"
#include "image_processor_service.pb.h"
#include "processor_api.h"

namespace {

// Singleton processor instances (initialized once at startup)
std::unique_ptr<jrb::infrastructure::cuda::GrayscaleProcessor> g_cuda_grayscale_processor;
std::unique_ptr<jrb::infrastructure::cpu::CpuGrayscaleProcessor> g_cpu_grayscale_processor;

// Helper to allocate response buffer (must be freed by caller with FreeResponse)
uint8_t* allocate_response(const std::string& serialized, int* out_len) {
    *out_len = static_cast<int>(serialized.size());
    uint8_t* buffer = new uint8_t[*out_len];
    std::memcpy(buffer, serialized.data(), *out_len);
    return buffer;
}

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

        // Create singleton processor instances (GPU and CPU)
        g_cuda_grayscale_processor =
            std::make_unique<jrb::infrastructure::cuda::GrayscaleProcessor>();
        g_cpu_grayscale_processor =
            std::make_unique<jrb::infrastructure::cpu::CpuGrayscaleProcessor>();

        init_resp.set_code(0);
        init_resp.set_message(
            "CUDA context, CPU processors, and telemetry initialized successfully");
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
        telemetry.CreateChildSpan("cgo-api", "ProcessImage", proc_req.trace_id(),
                                  proc_req.span_id(), static_cast<uint8_t>(proc_req.trace_flags()));
    jrb::core::telemetry::ScopedSpan scoped_span(span);

    // Check if processors are initialized
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
        jrb::ports::cgo::ImageBufferSource source(
            reinterpret_cast<const uint8_t*>(proc_req.image_data().data()), proc_req.width(),
            proc_req.height(), proc_req.channels());

        jrb::ports::cgo::ImageBufferSink sink;
        bool success = false;

        for (int i = 0; i < proc_req.filters_size(); i++) {
            cuda_learning::FilterType filter = proc_req.filters(i);

            if (filter == cuda_learning::FILTER_TYPE_NONE) {
                continue;
            }

            if (filter == cuda_learning::FILTER_TYPE_GRAYSCALE) {
                if (accelerator == cuda_learning::ACCELERATOR_TYPE_CUDA) {
                    scoped_span.AddEvent("Starting CUDA grayscale processing");
                    g_cuda_grayscale_processor->set_algorithm(
                        proto_to_cuda_algorithm(grayscale_type));
                    success = g_cuda_grayscale_processor->process(source, sink, "");
                    scoped_span.AddEvent("CUDA grayscale processing completed");
                } else {
                    scoped_span.AddEvent("Starting CPU grayscale processing");
                    g_cpu_grayscale_processor->set_algorithm(
                        proto_to_cpu_algorithm(grayscale_type));
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
    caps->set_api_version("2.0.0");
    caps->set_library_version("2.0.0");
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

void processor_free_response(uint8_t* ptr) {
    delete[] ptr;
}

}  // extern "C"

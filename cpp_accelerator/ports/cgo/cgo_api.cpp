#include "cpp_accelerator/ports/cgo/cgo_api.h"

#include <spdlog/spdlog.h>

#include <cstring>
#include <memory>

#include "cpp_accelerator/infrastructure/cuda/grayscale_processor.h"
#include "cpp_accelerator/ports/cgo/image_buffer_adapter.h"
#include "proto/image_processing.pb.h"

namespace {

// Singleton processor instance (initialized once at startup)
std::unique_ptr<jrb::infrastructure::cuda::GrayscaleProcessor> g_processor;

// Helper to allocate response buffer (must be freed by caller with FreeResponse)
uint8_t* allocate_response(const std::string& serialized, int* out_len) {
    *out_len = static_cast<int>(serialized.size());
    uint8_t* buffer = new uint8_t[*out_len];
    std::memcpy(buffer, serialized.data(), *out_len);
    return buffer;
}

}  // anonymous namespace

extern "C" {

bool CudaInit(const uint8_t* request, int request_len, uint8_t** response, int* response_len) {
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
        // Create singleton processor instance
        g_processor = std::make_unique<jrb::infrastructure::cuda::GrayscaleProcessor>();

        init_resp.set_code(0);
        init_resp.set_message("CUDA context initialized successfully");
        spdlog::info("CUDA initialization successful");
    } catch (const std::exception& e) {
        spdlog::error("CUDA initialization failed: {}", e.what());
        init_resp.set_code(2);
        init_resp.set_message(std::string("CUDA initialization failed: ") + e.what());
        *response = allocate_response(init_resp.SerializeAsString(), response_len);
        return false;
    }

    *response = allocate_response(init_resp.SerializeAsString(), response_len);
    return true;
}

void CudaCleanup() {
    spdlog::info("Cleaning up CUDA context");
    g_processor.reset();
}

bool ProcessImage(const uint8_t* request, int request_len, uint8_t** response, int* response_len) {
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

    // Check if processor is initialized
    if (!g_processor) {
        spdlog::error("CUDA processor not initialized. Call CudaInit first");
        proc_resp.set_code(3);
        proc_resp.set_message("CUDA processor not initialized. Call CudaInit first");
        *response = allocate_response(proc_resp.SerializeAsString(), response_len);
        return false;
    }

    // Validate filter type
    if (proc_req.filter() != cuda_learning::FILTER_TYPE_GRAYSCALE) {
        spdlog::error("Unsupported filter type: {}", proc_req.filter());
        proc_resp.set_code(4);
        proc_resp.set_message("Unsupported filter type");
        *response = allocate_response(proc_resp.SerializeAsString(), response_len);
        return false;
    }

    spdlog::debug("Processing image: {}x{}, {} channels", proc_req.width(), proc_req.height(),
                  proc_req.channels());

    try {
        // Create adapters
        jrb::ports::cgo::ImageBufferSource source(
            reinterpret_cast<const uint8_t*>(proc_req.image_data().data()), proc_req.width(),
            proc_req.height(), proc_req.channels());

        jrb::ports::cgo::ImageBufferSink sink;

        // Process image (filepath is ignored by our in-memory sink)
        bool success = g_processor->process(source, sink, "");

        if (!success) {
            spdlog::error("Image processing failed");
            proc_resp.set_code(5);
            proc_resp.set_message("Image processing failed");
            *response = allocate_response(proc_resp.SerializeAsString(), response_len);
            return false;
        }

        // Build successful response
        proc_resp.set_code(0);
        proc_resp.set_message("Image processed successfully");
        proc_resp.set_image_data(sink.get_data().data(), sink.get_data().size());
        proc_resp.set_width(sink.width());
        proc_resp.set_height(sink.height());
        proc_resp.set_channels(sink.channels());

        spdlog::debug("Image processing completed successfully");

    } catch (const std::exception& e) {
        spdlog::error("Exception during image processing: {}", e.what());
        proc_resp.set_code(6);
        proc_resp.set_message(std::string("Exception: ") + e.what());
        *response = allocate_response(proc_resp.SerializeAsString(), response_len);
        return false;
    }

    *response = allocate_response(proc_resp.SerializeAsString(), response_len);
    return true;
}

void FreeResponse(uint8_t* ptr) {
    delete[] ptr;
}

}  // extern "C"

#include <cstring>

#include "common.pb.h"
#include "image_processor_service.pb.h"
#include "processor_api.h"

namespace {

static bool initialized = false;

uint8_t* allocate_response(const std::string& serialized, int* out_len) {
    *out_len = static_cast<int>(serialized.size());
    uint8_t* buffer = new uint8_t[*out_len];
    std::memcpy(buffer, serialized.data(), *out_len);
    return buffer;
}

bool MockInit(const uint8_t* request, int request_len, uint8_t** response, int* response_len) {
    cuda_learning::InitRequest req;
    cuda_learning::InitResponse resp;

    if (!req.ParseFromArray(request, request_len)) {
        resp.set_code(1);
        resp.set_message("Failed to parse");
    } else {
        initialized = true;
        resp.set_code(0);
        resp.set_message("Mock processor initialized (passthrough mode)");
    }

    *response = allocate_response(resp.SerializeAsString(), response_len);
    return resp.code() == 0;
}

bool MockProcessImage(const uint8_t* request, int request_len, uint8_t** response,
                      int* response_len) {
    cuda_learning::ProcessImageRequest req;
    cuda_learning::ProcessImageResponse resp;

    if (!req.ParseFromArray(request, request_len)) {
        resp.set_code(1);
        resp.set_message("Failed to parse");
    } else {
        resp.set_code(0);
        resp.set_message("Mock passthrough (no processing)");
        resp.set_image_data(req.image_data());
        resp.set_width(req.width());
        resp.set_height(req.height());
        resp.set_channels(req.channels());
    }

    *response = allocate_response(resp.SerializeAsString(), response_len);
    return resp.code() == 0;
}

bool MockGetCapabilities(const uint8_t* request, int request_len, uint8_t** response,
                         int* response_len) {
    (void)request;
    (void)request_len;

    cuda_learning::GetCapabilitiesResponse resp;
    resp.set_code(0);
    resp.set_message("Mock capabilities");

    auto* caps = resp.mutable_capabilities();
    caps->set_api_version("2.0.0");
    caps->set_library_version("mock");
    caps->set_supports_streaming(false);
    caps->set_build_date(__DATE__);
    caps->set_build_commit("mock");

    auto* grayscale_filter = caps->add_filters();
    grayscale_filter->set_id("grayscale");
    grayscale_filter->set_name("Grayscale");
    grayscale_filter->add_supported_accelerators(cuda_learning::ACCELERATOR_TYPE_CPU);

    auto* algorithm_param = grayscale_filter->add_parameters();
    algorithm_param->set_id("algorithm");
    algorithm_param->set_name("Algorithm");
    algorithm_param->set_type("select");
    algorithm_param->add_options("bt601");
    algorithm_param->add_options("bt709");
    algorithm_param->set_default_value("bt601");

    *response = allocate_response(resp.SerializeAsString(), response_len);
    return true;
}

}  // namespace

extern "C" {

processor_version_t processor_api_version(void) {
    processor_version_t version;
    version.major = (PROCESSOR_API_VERNUM >> 16) & 0xFF;
    version.minor = (PROCESSOR_API_VERNUM >> 8) & 0xFF;
    version.patch = PROCESSOR_API_VERNUM & 0xFF;
    return version;
}

bool processor_init(const uint8_t* request_buf, int request_len, uint8_t** response_buf,
                    int* response_len) {
    return MockInit(request_buf, request_len, response_buf, response_len);
}

void processor_cleanup() {
    initialized = false;
}

bool processor_process_image(const uint8_t* request_buf, int request_len, uint8_t** response_buf,
                             int* response_len) {
    return MockProcessImage(request_buf, request_len, response_buf, response_len);
}

bool processor_get_capabilities(const uint8_t* request_buf, int request_len, uint8_t** response_buf,
                                int* response_len) {
    return MockGetCapabilities(request_buf, request_len, response_buf, response_len);
}

void processor_free_response(uint8_t* buf) {
    delete[] buf;
}
}

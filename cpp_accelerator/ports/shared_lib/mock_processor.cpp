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
    caps->set_api_version("1.0.0");
    caps->set_library_version("mock");
    caps->add_supported_filters("none");
    caps->add_supported_accelerators("mock");
    caps->set_supports_streaming(false);
    caps->set_build_date(__DATE__);
    caps->set_build_commit("mock");

    *response = allocate_response(resp.SerializeAsString(), response_len);
    return true;
}

}  // namespace

extern "C" {

processor_version_t processor_api_version(void) {
    processor_version_t version;
    version.major = 1;
    version.minor = 0;
    version.patch = 0;
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

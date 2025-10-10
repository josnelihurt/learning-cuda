#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize CUDA context (call once at startup)
// Returns: true on success, false on failure
// request: serialized InitRequest protobuf
// request_len: length of request in bytes
// response: output pointer to serialized InitResponse (caller must free with FreeResponse)
// response_len: output length of response
bool CudaInit(const uint8_t* request, int request_len, uint8_t** response, int* response_len);

// Cleanup CUDA context (call at shutdown)
void CudaCleanup();

// Process an image with specified filter
// Returns: true on success, false on failure
// request: serialized ProcessImageRequest protobuf
// request_len: length of request in bytes
// response: output pointer to serialized ProcessImageResponse (caller must free with FreeResponse)
// response_len: output length of response
bool ProcessImage(const uint8_t* request, int request_len, uint8_t** response, int* response_len);

// Free memory allocated by C++ (must be called on all responses)
void FreeResponse(uint8_t* ptr);

#ifdef __cplusplus
}
#endif

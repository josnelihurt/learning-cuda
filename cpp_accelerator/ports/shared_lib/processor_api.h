/**
 * @file processor_api.h
 * @brief Public C API for CUDA image processor shared library
 *
 * This header defines the stable ABI for the CUDA image processor library.
 * The API uses protobuf serialized messages for all data exchange, enabling
 * language-agnostic integration and version compatibility.
 *
 * Memory Management:
 * - Response buffers are allocated by the library and must be freed using
 *   processor_free_response() to avoid memory leaks.
 * - Request buffers are managed by the caller.
 *
 * Thread Safety:
 * - processor_init() and processor_cleanup() are not thread-safe.
 * - processor_process_image() and processor_get_capabilities() can be called
 *   concurrently after successful initialization.
 *
 * Error Handling:
 * - Functions return false on failure, true on success.
 * - Detailed error information is available in the response protobuf message.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Library version as a string in semantic versioning format (MAJOR.MINOR.PATCH)
 */
#define PROCESSOR_API_VERSION "2.0.0"

/**
 * Library version as a single hex number: 0xMMNNPP
 * MM = major, NN = minor, PP = patch
 * Used for compile-time version checks
 */
#define PROCESSOR_API_VERNUM 0x020000

/**
 * @brief Version information structure
 */
typedef struct {
    int major; /**< Major version number (breaking changes) */
    int minor; /**< Minor version number (backward-compatible features) */
    int patch; /**< Patch version number (backward-compatible fixes) */
} processor_version_t;

/**
 * @brief Get the API version of the loaded library
 *
 * This function retrieves the version information from the library at runtime.
 * Use this to verify compatibility between the loader and the loaded library.
 * The version follows semantic versioning: major version must match for compatibility.
 *
 * @return Version structure populated with major, minor, and patch numbers
 *
 * @note This function is always safe to call and never fails.
 * @see PROCESSOR_API_VERSION for compile-time version string
 * @see PROCESSOR_API_VERNUM for compile-time version number
 */
processor_version_t processor_api_version(void);

/**
 * @brief Initialize the processor library
 *
 * Initializes CUDA context, allocates GPU resources, and prepares the processor
 * for image processing operations. Must be called successfully before any other
 * processing functions.
 *
 * @param request_buf Pointer to serialized InitRequest protobuf message
 * @param request_len Length of the request buffer in bytes
 * @param response_buf Output pointer for serialized InitResponse protobuf message
 * @param response_len Output pointer for response buffer length in bytes
 *
 * @return true on successful initialization, false on failure
 *
 * @note The response buffer is allocated by the library and must be freed using
 *       processor_free_response() after use.
 * @note This function is not thread-safe. Call only once during application startup.
 * @warning Calling this function multiple times without cleanup leads to resource leaks.
 */
bool processor_init(const uint8_t* request_buf, int request_len, uint8_t** response_buf,
                    int* response_len);

/**
 * @brief Clean up and release all processor resources
 *
 * Releases CUDA context, deallocates GPU memory, and cleans up all resources
 * allocated during initialization. Should be called during application shutdown.
 *
 * @note This function is not thread-safe. Ensure all processing operations have
 *       completed before calling.
 * @note After calling this function, processor_init() must be called again before
 *       any processing operations.
 * @warning Do not call while image processing operations are in progress.
 */
void processor_cleanup();

/**
 * @brief Process an image using configured filters and accelerators
 *
 * Applies the requested image processing filters (grayscale, blur, etc.) using
 * the specified accelerator (GPU or CPU). The operation is performed asynchronously
 * on the GPU when GPU acceleration is requested.
 *
 * @param request_buf Pointer to serialized ProcessImageRequest protobuf message
 * @param request_len Length of the request buffer in bytes
 * @param response_buf Output pointer for serialized ProcessImageResponse protobuf message
 * @param response_len Output pointer for response buffer length in bytes
 *
 * @return true on successful processing, false on failure
 *
 * @note The response buffer is allocated by the library and must be freed using
 *       processor_free_response() after use.
 * @note This function is thread-safe after successful initialization.
 * @note The input image data is not modified. Output is a new buffer in the response.
 */
bool processor_process_image(const uint8_t* request_buf, int request_len, uint8_t** response_buf,
                             int* response_len);

/**
 * @brief Query the library's processing capabilities
 *
 * Returns information about supported filters, accelerators, algorithms, and
 * library metadata. Useful for feature discovery and validation.
 *
 * @param request_buf Pointer to serialized GetCapabilitiesRequest protobuf message
 * @param request_len Length of the request buffer in bytes
 * @param response_buf Output pointer for serialized GetCapabilitiesResponse protobuf message
 * @param response_len Output pointer for response buffer length in bytes
 *
 * @return true on success, false on failure
 *
 * @note The response buffer is allocated by the library and must be freed using
 *       processor_free_response() after use.
 * @note This function is thread-safe and can be called without prior initialization.
 * @note Capabilities are static and do not change during the library's lifetime.
 */
bool processor_get_capabilities(const uint8_t* request_buf, int request_len, uint8_t** response_buf,
                                int* response_len);

/**
 * @brief Free a response buffer allocated by the library
 *
 * Deallocates memory for response buffers returned by processor_init(),
 * processor_process_image(), or processor_get_capabilities().
 *
 * @param buf Pointer to the response buffer to free
 *
 * @note Always call this function for every non-NULL response buffer to prevent
 *       memory leaks.
 * @note Passing NULL is safe and will be ignored.
 * @warning Do not call free() or delete directly on response buffers.
 * @warning Do not use the buffer after calling this function.
 */
void processor_free_response(uint8_t* buf);

#ifdef __cplusplus
}
#endif

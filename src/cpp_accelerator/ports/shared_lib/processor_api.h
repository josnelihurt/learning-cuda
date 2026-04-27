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
#define PROCESSOR_API_VERSION "2.1.0"

/**
 * Library version as a single hex number: 0xMMNNPP
 * MM = major, NN = minor, PP = patch
 * Used for compile-time version checks
 */
#define PROCESSOR_API_VERNUM 0x020100

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

#ifdef __cplusplus
}
#endif

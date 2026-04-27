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

#ifdef __cplusplus
}
#endif

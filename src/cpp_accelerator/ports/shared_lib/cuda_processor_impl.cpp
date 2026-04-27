#include <cstdlib>
#include <cstring>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic pop

#include "src/cpp_accelerator/ports/shared_lib/processor_api.h"

extern "C" {

processor_version_t processor_api_version(void) {
  processor_version_t version;
  version.major = (PROCESSOR_API_VERNUM >> 16) & 0xFF;
  version.minor = (PROCESSOR_API_VERNUM >> 8) & 0xFF;
  version.patch = PROCESSOR_API_VERNUM & 0xFF;
  return version;
}

}  // extern "C"

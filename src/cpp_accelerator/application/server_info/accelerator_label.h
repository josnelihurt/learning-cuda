#pragma once

#include <string>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"

namespace jrb::application::server_info {

// Human-readable name for an AcceleratorType — used in the
// GetAcceleratorCapabilities response for UI display.
std::string AcceleratorTypeLabel(cuda_learning::AcceleratorType type);

}  // namespace jrb::application::server_info

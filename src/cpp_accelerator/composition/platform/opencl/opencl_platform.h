#pragma once

#include "src/cpp_accelerator/application/engine/filter_factory_registry.h"

namespace jrb::application::engine::platform::opencl {

void RegisterFactories(FilterFactoryRegistry& registry);

}  // namespace jrb::application::engine::platform::opencl

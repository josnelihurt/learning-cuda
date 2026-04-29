#include "src/cpp_accelerator/application/engine/filter_factory_registry.h"

namespace jrb::application::engine {

void FilterFactoryRegistry::Register(std::unique_ptr<IFilterFactory> factory) {
  int key = static_cast<int>(factory->GetAcceleratorType());
  factories_[key] = std::move(factory);
}

IFilterFactory* FilterFactoryRegistry::GetFactory(cuda_learning::AcceleratorType type) const {
  auto it = factories_.find(static_cast<int>(type));
  if (it == factories_.end()) return nullptr;
  return it->second.get();
}

std::vector<cuda_learning::AcceleratorType> FilterFactoryRegistry::GetRegisteredTypes() const {
  std::vector<cuda_learning::AcceleratorType> types;
  types.reserve(factories_.size());
  for (const auto& [key, _] : factories_) {
    types.push_back(static_cast<cuda_learning::AcceleratorType>(key));
  }
  return types;
}

}  // namespace jrb::application::engine

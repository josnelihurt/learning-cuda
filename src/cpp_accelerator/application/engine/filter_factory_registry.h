#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "src/cpp_accelerator/application/engine/i_filter_factory.h"

namespace jrb::application::engine {

class FilterFactoryRegistry {
 public:
  FilterFactoryRegistry() = default;

  // Registers a factory. Only one factory per AcceleratorType is allowed;
  // a second registration for the same type replaces the first.
  void Register(std::unique_ptr<IFilterFactory> factory);

  // Returns the factory for the given accelerator type, or nullptr if not registered.
  IFilterFactory* GetFactory(cuda_learning::AcceleratorType type) const;

  // Returns all registered accelerator types.
  std::vector<cuda_learning::AcceleratorType> GetRegisteredTypes() const;

 private:
  std::unordered_map<int, std::unique_ptr<IFilterFactory>> factories_;
};

}  // namespace jrb::application::engine

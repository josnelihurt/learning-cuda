#pragma once

#include <memory>
#include <vector>

#include "proto/_virtual_imports/common_proto/common.pb.h"
#include "src/cpp_accelerator/application/engine/filter_descriptor.h"
#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"

namespace jrb::application::engine {

class IFilterFactory {
 public:
  virtual ~IFilterFactory() = default;

  virtual cuda_learning::AcceleratorType GetAcceleratorType() const = 0;

  // Returns the filter definitions (with parameters) this factory supports.
  virtual std::vector<FilterDescriptor> GetFilterDescriptors() const = 0;

  // Creates a filter instance for the given type with the supplied params.
  // Returns nullptr if the type is not supported by this factory.
  virtual std::unique_ptr<jrb::domain::interfaces::IFilter> CreateFilter(
      jrb::domain::interfaces::FilterType type, const FilterCreationParams& params) const = 0;
};

}  // namespace jrb::application::engine

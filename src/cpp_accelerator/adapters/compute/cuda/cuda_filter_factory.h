#pragma once

#include "src/cpp_accelerator/application/engine/i_filter_factory.h"

namespace jrb::infrastructure::cuda {

class CudaFilterFactory : public jrb::application::engine::IFilterFactory {
 public:
  CudaFilterFactory() = default;
  ~CudaFilterFactory() override = default;

  cuda_learning::AcceleratorType GetAcceleratorType() const override;

  std::vector<jrb::application::engine::FilterDescriptor> GetFilterDescriptors() const override;

  std::unique_ptr<jrb::domain::interfaces::IFilter> CreateFilter(
      jrb::domain::interfaces::FilterType type,
      const jrb::application::engine::FilterCreationParams& params) const override;
};

}  // namespace jrb::infrastructure::cuda

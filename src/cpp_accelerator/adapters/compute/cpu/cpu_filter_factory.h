#pragma once

#include "src/cpp_accelerator/application/engine/i_filter_factory.h"

namespace jrb::infrastructure::cpu {

class CpuFilterFactory : public jrb::application::engine::IFilterFactory {
 public:
  CpuFilterFactory() = default;
  ~CpuFilterFactory() override = default;

  cuda_learning::AcceleratorType GetAcceleratorType() const override;

  std::vector<jrb::application::engine::FilterDescriptor> GetFilterDescriptors() const override;

  std::unique_ptr<jrb::domain::interfaces::IFilter> CreateFilter(
      jrb::domain::interfaces::FilterType type,
      const jrb::application::engine::FilterCreationParams& params) const override;
};

}  // namespace jrb::infrastructure::cpu

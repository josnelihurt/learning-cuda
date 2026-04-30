#pragma once

#include "src/cpp_accelerator/application/engine/i_filter_factory.h"

namespace jrb::adapters::compute::opencl {

// Filter factory for the OpenCL accelerator.
// Supports grayscale and Gaussian blur. No user-configurable parameters —
// filters apply with fixed defaults (BT.601 grayscale, 5×5 Gaussian blur σ=1).
class OpenCLFilterFactory : public jrb::application::engine::IFilterFactory {
 public:
  OpenCLFilterFactory() = default;
  ~OpenCLFilterFactory() override = default;

  cuda_learning::AcceleratorType GetAcceleratorType() const override;

  std::vector<jrb::application::engine::FilterDescriptor> GetFilterDescriptors() const override;

  std::unique_ptr<jrb::domain::interfaces::IFilter> CreateFilter(
      jrb::domain::interfaces::FilterType type,
      const jrb::application::engine::FilterCreationParams& params) const override;
};

}  // namespace jrb::adapters::compute::opencl

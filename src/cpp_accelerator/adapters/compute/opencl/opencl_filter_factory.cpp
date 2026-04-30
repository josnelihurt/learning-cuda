#include "src/cpp_accelerator/adapters/compute/opencl/opencl_filter_factory.h"

#include "src/cpp_accelerator/adapters/compute/opencl/filters/blur_filter.h"
#include "src/cpp_accelerator/adapters/compute/opencl/filters/grayscale_filter.h"
#include "src/cpp_accelerator/application/engine/filter_creation_dispatch.hpp"
#include "src/cpp_accelerator/application/engine/filter_descriptor.h"

namespace jrb::adapters::compute::opencl {

using jrb::application::engine::FilterCreationParams;
using jrb::application::engine::FilterDescriptor;
using jrb::domain::interfaces::FilterType;

cuda_learning::AcceleratorType OpenCLFilterFactory::GetAcceleratorType() const {
  return cuda_learning::ACCELERATOR_TYPE_OPENCL;
}

std::vector<FilterDescriptor> OpenCLFilterFactory::GetFilterDescriptors() const {
  // OpenCL filters have no user-configurable parameters.
  // They apply with sensible built-in defaults (BT.601 grayscale, 5×5 Gaussian blur).
  // No parameters intentionally — OpenCL applies BT.601 with no configuration.
  // No parameters intentionally — OpenCL applies a fixed 5×5 σ=1.0 kernel.
  return {
      FilterDescriptor{.id = "grayscale", .name = "Grayscale", .parameters = {}},
      FilterDescriptor{.id = "blur", .name = "Gaussian Blur", .parameters = {}},
  };
}

std::unique_ptr<jrb::domain::interfaces::IFilter> OpenCLFilterFactory::CreateFilter(
    FilterType type, const FilterCreationParams& /*params*/) const {
  return jrb::application::engine::DispatchCreateFilter(
      type, []() { return std::make_unique<GrayscaleFilter>(); },
      []() { return std::make_unique<GaussianBlurFilter>(); });
}

}  // namespace jrb::adapters::compute::opencl

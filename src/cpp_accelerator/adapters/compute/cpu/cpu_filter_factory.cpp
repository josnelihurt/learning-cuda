#include "src/cpp_accelerator/adapters/compute/cpu/cpu_filter_factory.h"

#include "src/cpp_accelerator/adapters/compute/cpu/blur_filter.h"
#include "src/cpp_accelerator/adapters/compute/cpu/grayscale_filter.h"
#include "src/cpp_accelerator/application/engine/filter_creation_dispatch.hpp"
#include "src/cpp_accelerator/application/engine/filter_descriptor.h"

namespace jrb::infrastructure::cpu {

using jrb::application::engine::BlurBorderMode;
using jrb::application::engine::FilterCreationParams;
using jrb::application::engine::FilterDescriptor;
using jrb::application::engine::ParameterDescriptor;
using jrb::application::engine::ParameterOption;
using jrb::domain::interfaces::FilterType;

cuda_learning::AcceleratorType CpuFilterFactory::GetAcceleratorType() const {
  return cuda_learning::ACCELERATOR_TYPE_CPU;
}

std::vector<FilterDescriptor> CpuFilterFactory::GetFilterDescriptors() const {
  std::vector<FilterDescriptor> descriptors;

  descriptors.emplace_back(FilterDescriptor{
      .id = "grayscale",
      .name = "Grayscale",
      .parameters = {ParameterDescriptor{
          .id = "algorithm",
          .name = "Algorithm",
          .type = "select",
          .default_value = "bt601",
          .options = {{"bt601", "bt601"},
                      {"bt709", "bt709"},
                      {"average", "average"},
                      {"lightness", "lightness"},
                      {"luminosity", "luminosity"}},
          .metadata = {{"required", "true"}, {"min_items", "1"}, {"max_items", "1"}},
      }},
  });

  descriptors.emplace_back(FilterDescriptor{
      .id = "blur",
      .name = "Gaussian Blur",
      .parameters =
          {
              ParameterDescriptor{
                  .id = "kernel_size",
                  .name = "Kernel Size",
                  .type = "range",
                  .default_value = "5",
                  .options = {},
                  .metadata = {{"required", "true"}, {"min", "1"}, {"max", "31"}, {"step", "2"}},
              },
              ParameterDescriptor{
                  .id = "sigma",
                  .name = "Sigma",
                  .type = "number",
                  .default_value = "1.0",
                  .options = {},
                  .metadata = {{"required", "true"}, {"min", "0"}, {"max", "100"}, {"step", "0.1"}},
              },
              ParameterDescriptor{
                  .id = "border_mode",
                  .name = "Border Mode",
                  .type = "select",
                  .default_value = "REFLECT",
                  .options = {{"CLAMP", "CLAMP"}, {"REFLECT", "REFLECT"}, {"WRAP", "WRAP"}},
                  .metadata = {{"required", "true"}, {"min_items", "1"}, {"max_items", "1"}},
              },
              ParameterDescriptor{
                  .id = "separable",
                  .name = "Separable",
                  .type = "checkbox",
                  .default_value = "true",
                  .options = {},
                  .metadata = {{"required", "true"}},
              },
          },
  });

  return descriptors;
}

std::unique_ptr<jrb::domain::interfaces::IFilter> CpuFilterFactory::CreateFilter(
    FilterType type, const FilterCreationParams& params) const {
  return jrb::application::engine::DispatchCreateFilter(
      type, [&params]() { return std::make_unique<GrayscaleFilter>(params.grayscale_algorithm); },
      [&params]() {
        BorderMode border_mode = BorderMode::kReflect;
        switch (params.blur_border_mode) {
          case BlurBorderMode::CLAMP:
            border_mode = BorderMode::kClamp;
            break;
          case BlurBorderMode::REFLECT:
            border_mode = BorderMode::kReflect;
            break;
          case BlurBorderMode::WRAP:
            border_mode = BorderMode::kWrap;
            break;
        }
        return std::make_unique<GaussianBlurFilter>(params.blur_kernel_size, params.blur_sigma,
                                                    border_mode, params.blur_separable);
      });
}

}  // namespace jrb::infrastructure::cpu

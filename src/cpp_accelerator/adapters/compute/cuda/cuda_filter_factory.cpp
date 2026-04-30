#include "src/cpp_accelerator/adapters/compute/cuda/cuda_filter_factory.h"

#include "src/cpp_accelerator/adapters/compute/cuda/filters/blur_filter.h"
#include "src/cpp_accelerator/adapters/compute/cuda/filters/grayscale_filter.h"
#include "src/cpp_accelerator/application/engine/filter_creation_dispatch.hpp"
#include "src/cpp_accelerator/application/engine/filter_descriptor.h"

namespace jrb::infrastructure::cuda {

using jrb::application::engine::BlurBorderMode;
using jrb::application::engine::FilterCreationParams;
using jrb::application::engine::FilterDescriptor;
using jrb::application::engine::ParameterDescriptor;
using jrb::application::engine::ParameterOption;
using jrb::domain::interfaces::FilterType;

cuda_learning::AcceleratorType CudaFilterFactory::GetAcceleratorType() const {
  return cuda_learning::ACCELERATOR_TYPE_CUDA;
}

std::vector<FilterDescriptor> CudaFilterFactory::GetFilterDescriptors() const {
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

  descriptors.emplace_back(FilterDescriptor{
      .id = "model_inference",
      .name = "Model Inference",
      .parameters =
          {
              ParameterDescriptor{
                  .id = "model_id",
                  .name = "Model",
                  .type = "select",
                  .default_value = "yolov10n",
                  .options = {{"yolov10n", "yolov10n"}},
                  .metadata = {{"required", "true"}, {"min_items", "1"}, {"max_items", "1"}},
              },
              ParameterDescriptor{
                  .id = "confidence_threshold",
                  .name = "Confidence Threshold",
                  .type = "number",
                  .default_value = "0.45",
                  .options = {},
                  .metadata = {{"required", "true"}, {"min", "0"}, {"max", "1"}, {"step", "0.01"}},
              },
          },
  });

  return descriptors;
}

std::unique_ptr<jrb::domain::interfaces::IFilter> CudaFilterFactory::CreateFilter(
    FilterType type, const FilterCreationParams& params) const {
  return jrb::application::engine::DispatchCreateFilter(
      type,
      [&params]() { return std::make_unique<GrayscaleFilter>(params.grayscale_algorithm); },
      [&params]() {
        BorderMode border_mode = BorderMode::REFLECT;
        switch (params.blur_border_mode) {
          case BlurBorderMode::CLAMP:
            border_mode = BorderMode::CLAMP;
            break;
          case BlurBorderMode::REFLECT:
            border_mode = BorderMode::REFLECT;
            break;
          case BlurBorderMode::WRAP:
            border_mode = BorderMode::WRAP;
            break;
        }
        return std::make_unique<CudaGaussianBlurFilter>(params.blur_kernel_size, params.blur_sigma,
                                                        border_mode, params.blur_separable);
      });
}

}  // namespace jrb::infrastructure::cuda

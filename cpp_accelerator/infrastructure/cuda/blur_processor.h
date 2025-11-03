#pragma once

#include <cstdint>
#include <vector>

#include "cpp_accelerator/domain/interfaces/filters/i_filter.h"

namespace jrb::infrastructure::cuda {

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::FilterType;
using jrb::domain::interfaces::IFilter;

enum class BorderMode : std::uint8_t { CLAMP, REFLECT, WRAP };

class CudaGaussianBlurFilter : public IFilter {
public:
  explicit CudaGaussianBlurFilter(int kernel_size = 5, float sigma = 1.0F,
                                  BorderMode border_mode = BorderMode::REFLECT,
                                  bool separable = true);
  ~CudaGaussianBlurFilter() override;

  bool Apply(FilterContext& context) override;

  FilterType GetType() const override;

  bool IsInPlace() const override;

  void SetKernelSize(int size);
  int GetKernelSize() const;

  void SetSigma(float sigma);
  float GetSigma() const;

  void SetBorderMode(BorderMode mode);
  BorderMode GetBorderMode() const;

private:
  void InitializeKernel();

  int kernel_size_;
  float sigma_;
  BorderMode border_mode_;
  bool separable_;
  std::vector<float> kernel_;
};

}  // namespace jrb::infrastructure::cuda

#pragma once

#include <CL/cl.h>

#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"

namespace jrb::infrastructure::opencl {

// Applies a separable 5×5 Gaussian blur (σ = 1.0) via two OpenCL passes:
// horizontal then vertical. Input and output are packed RGB (3 bytes/pixel).
class OpenCLBlurFilter : public jrb::domain::interfaces::IFilter {
 public:
  OpenCLBlurFilter();
  ~OpenCLBlurFilter() override;

  bool Apply(jrb::domain::interfaces::FilterContext& context) override;
  jrb::domain::interfaces::FilterType GetType() const override;
  bool IsInPlace() const override;

 private:
  bool EnsureKernels();

  bool kernel_ready_;
  cl_program program_;
  cl_kernel kernel_h_;  // horizontal pass
  cl_kernel kernel_v_;  // vertical pass
};

}  // namespace jrb::infrastructure::opencl

#pragma once

#include <CL/cl.h>

#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"

namespace jrb::infrastructure::opencl {

// Converts an RGB image to single-channel grayscale using the BT.601 luma
// coefficients (L = 0.299R + 0.587G + 0.114B) on the OpenCL device.
class OpenCLGrayscaleFilter : public jrb::domain::interfaces::IFilter {
 public:
  OpenCLGrayscaleFilter();
  ~OpenCLGrayscaleFilter() override;

  bool Apply(jrb::domain::interfaces::FilterContext& context) override;
  jrb::domain::interfaces::FilterType GetType() const override;
  bool IsInPlace() const override;

 private:
  bool EnsureKernel();

  bool kernel_ready_;
  cl_program program_;
  cl_kernel kernel_;
};

}  // namespace jrb::infrastructure::opencl

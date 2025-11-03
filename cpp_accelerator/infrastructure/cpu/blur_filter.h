#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "cpp_accelerator/domain/interfaces/i_pixel_getter.h"
#include "cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::infrastructure::cpu {
namespace internal {
class BorderClamperFactory;
}

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::FilterType;
using jrb::domain::interfaces::IFilter;
using jrb::domain::interfaces::ImageBuffer;
using jrb::domain::interfaces::ImageBufferMut;
using jrb::domain::interfaces::IPixelGetter;

enum class BorderMode : std::uint8_t { CLAMP, REFLECT, WRAP };

class GaussianBlurFilter : public IFilter, public IPixelGetter {
public:
  explicit GaussianBlurFilter(int kernel_size = 5, float sigma = 1.0F,
                              BorderMode border_mode = BorderMode::REFLECT, bool separable = true);
  ~GaussianBlurFilter() override;

  /**
   * @brief Applies Gaussian blur filter to the input image
   *
   * Implements two optimization strategies:
   * - Separable convolution (default): Faster O(k) complexity by applying 1D blur
   *   horizontally then vertically. Requires temporary buffer for intermediate results.
   * - Full 2D convolution: Slower O(k^2) complexity but no extra memory allocation.
   *
   * @param context Filter context containing input and output image buffers
   * @return true on success, false on failure
   *
   * @note When separable=true, temporary buffer allocation is required for the
   *       intermediate result between horizontal and vertical passes. This is a
   *       fundamental requirement of separable convolution algorithms where we
   *       cannot read and write to the same buffer simultaneously.
   */
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
  float GetPixelValue(const ImageBuffer& buffer, int x, int y) const override;
  void ApplyHorizontalBlur(const ImageBuffer& input, ImageBufferMut& output);
  void ApplyVerticalBlur(const ImageBuffer& input, ImageBufferMut& output);
  void ApplyFullBlur(const ImageBuffer& input, ImageBufferMut& output);

  int kernel_size_;
  float sigma_;
  BorderMode border_mode_;
  bool separable_;
  std::vector<float> kernel_;
  std::unique_ptr<internal::BorderClamperFactory> border_clamper_factory_;
};

}  // namespace jrb::infrastructure::cpu

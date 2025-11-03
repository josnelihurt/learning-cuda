#include "cpp_accelerator/infrastructure/cpu/blur_filter.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include "cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "cpp_accelerator/domain/interfaces/i_pixel_getter.h"
#include "cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::infrastructure::cpu {

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::FilterType;
using jrb::domain::interfaces::IFilter;
using jrb::domain::interfaces::ImageBuffer;
using jrb::domain::interfaces::ImageBufferMut;
using jrb::domain::interfaces::IPixelGetter;

constexpr float SQRT_2PI = 2.506628274631000242F;

namespace {
constexpr float LUMA_RED_WEIGHT_BT601 = 0.299F;
constexpr float LUMA_GREEN_WEIGHT_BT601 = 0.587F;
constexpr float LUMA_BLUE_WEIGHT_BT601 = 0.114F;

enum class BlurDirection { HORIZONTAL, VERTICAL };

float Compute1DConvolution(const ImageBuffer& buffer, int center_x, int center_y,
                           const std::vector<float>& kernel, int radius,
                           const IPixelGetter& pixel_getter, BlurDirection direction) {
  float sum = 0.0F;

  for (int k = -radius; k <= radius; k++) {
    int pixel_x = (direction == BlurDirection::HORIZONTAL) ? center_x + k : center_x;
    int pixel_y = (direction == BlurDirection::HORIZONTAL) ? center_y : center_y + k;
    float weight = kernel[k + radius];
    sum += pixel_getter.GetPixelValue(buffer, pixel_x, pixel_y) * weight;
  }

  return sum;
}

float Compute2DConvolution(const ImageBuffer& buffer, int center_x, int center_y,
                           const std::vector<float>& kernel, int radius,
                           const IPixelGetter& pixel_getter) {
  float sum = 0.0F;

  for (int ky = -radius; ky <= radius; ky++) {
    for (int kx = -radius; kx <= radius; kx++) {
      float weight_y = kernel[ky + radius];
      float weight_x = kernel[kx + radius];
      float weight = weight_x * weight_y;
      sum += pixel_getter.GetPixelValue(buffer, center_x + kx, center_y + ky) * weight;
    }
  }

  return sum;
}

void WritePixelToOutput(ImageBufferMut& output, int x, int y, float blurred_value) {
  int output_idx = (y * output.width + x) * output.channels;
  unsigned char value = static_cast<unsigned char>(std::clamp(blurred_value, 0.0F, 255.0F));
  output.data[output_idx] = value;
  if (output.channels >= 3) {
    output.data[output_idx + 1] = value;
    output.data[output_idx + 2] = value;
  }
}

template <BlurDirection Direction>
void Apply1DBlur(const ImageBuffer& input, ImageBufferMut& output, const std::vector<float>& kernel,
                 int radius, const IPixelGetter& pixel_getter) {
  for (int y = 0; y < input.height; y++) {
    for (int x = 0; x < input.width; x++) {
      float sum = Compute1DConvolution(input, x, y, kernel, radius, pixel_getter, Direction);
      WritePixelToOutput(output, x, y, sum);
    }
  }
}

std::vector<float> GenerateGaussianKernel(int size, float sigma) {
  if (size % 2 == 0) {
    size++;
  }

  int radius = size / 2;
  std::vector<float> kernel(size);

  float sum = 0.0F;
  for (int i = 0; i < size; i++) {
    float x = static_cast<float>(i - radius);
    float value = std::exp(-0.5F * (x * x) / (sigma * sigma)) / (sigma * SQRT_2PI);
    kernel[i] = value;
    sum += value;
  }

  for (int i = 0; i < size; i++) {
    kernel[i] /= sum;
  }

  return kernel;
}

}  // anonymous namespace

namespace internal {

/**
 * @brief Interface defining border extension strategies for Gaussian blur convolution
 *
 * Gaussian blur implements discrete convolution where each output pixel is computed
 * as a weighted average of neighboring input pixels using a Gaussian kernel. When the
 * kernel extends beyond image boundaries, we must define how to handle out-of-bounds
 * pixel access. This interface abstracts different boundary extension methods from
 * image processing literature:
 *
 * - Zero-padding: Assumes zero intensity outside boundaries
 * - Clamp: Extends the nearest edge pixel (replication)
 * - Reflect: Mirrors pixels across boundaries (symmetric extension)
 * - Wrap: Tiles the image periodically (circular extension)
 *
 * The choice of boundary method affects visual artifacts near edges. Reflect and wrap
 * provide smoother transitions for gradient-based filters like Gaussian blur, while
 * clamp is simpler but may create edge discontinuities.
 */
class IBorderClamper {
public:
  virtual ~IBorderClamper() = default;
  virtual int ClampX(int x, int width) const = 0;
  virtual int ClampY(int y, int height) const = 0;
};

class ClampBorder : public IBorderClamper {
public:
  int ClampX(int x, int width) const override { return std::clamp(x, 0, width - 1); }
  int ClampY(int y, int height) const override { return std::clamp(y, 0, height - 1); }
};

class ReflectBorder : public IBorderClamper {
public:
  int ClampX(int x, int width) const override {
    if (x < 0) {
      return -x - 1;
    }
    if (x >= width) {
      return 2 * width - x - 1;
    }
    return x;
  }
  int ClampY(int y, int height) const override {
    if (y < 0) {
      return -y - 1;
    }
    if (y >= height) {
      return 2 * height - y - 1;
    }
    return y;
  }
};

class WrapBorder : public IBorderClamper {
public:
  int ClampX(int x, int width) const override { return ((x % width) + width) % width; }
  int ClampY(int y, int height) const override { return ((y % height) + height) % height; }
};

class BorderClamperFactory {
public:
  const IBorderClamper& Get(BorderMode mode) const {
    switch (mode) {
      case BorderMode::CLAMP:
        return clamp_border_;
      case BorderMode::REFLECT:
        return reflect_border_;
      case BorderMode::WRAP:
        return wrap_border_;
      default:
        return clamp_border_;
    }
  }

private:
  ClampBorder clamp_border_;
  ReflectBorder reflect_border_;
  WrapBorder wrap_border_;
};
}  // namespace internal

GaussianBlurFilter::GaussianBlurFilter(int kernel_size, float sigma, BorderMode border_mode,
                                       bool separable)
    : kernel_size_(kernel_size), sigma_(sigma), border_mode_(border_mode), separable_(separable) {
  kernel_ = GenerateGaussianKernel(kernel_size_, sigma_);
  border_clamper_factory_ = std::make_unique<internal::BorderClamperFactory>();
}

GaussianBlurFilter::~GaussianBlurFilter() = default;

void GaussianBlurFilter::SetKernelSize(int size) {
  kernel_size_ = size;
  kernel_ = GenerateGaussianKernel(kernel_size_, sigma_);
}

void GaussianBlurFilter::SetSigma(float sigma) {
  sigma_ = sigma;
  kernel_ = GenerateGaussianKernel(kernel_size_, sigma_);
}

void GaussianBlurFilter::SetBorderMode(BorderMode mode) {
  border_mode_ = mode;
}

FilterType GaussianBlurFilter::GetType() const {
  return FilterType::BLUR;
}

bool GaussianBlurFilter::IsInPlace() const {
  return false;
}

int GaussianBlurFilter::GetKernelSize() const {
  return kernel_size_;
}

float GaussianBlurFilter::GetSigma() const {
  return sigma_;
}

BorderMode GaussianBlurFilter::GetBorderMode() const {
  return border_mode_;
}

float GaussianBlurFilter::GetPixelValue(const ImageBuffer& buffer, int x, int y) const {
  const internal::IBorderClamper& clamper = border_clamper_factory_->Get(border_mode_);

  int clamped_x = clamper.ClampX(x, buffer.width);
  int clamped_y = clamper.ClampY(y, buffer.height);

  int index = (clamped_y * buffer.width + clamped_x) * buffer.channels;
  if (buffer.channels >= 3) {
    float r = static_cast<float>(buffer.data[index]);
    float g = static_cast<float>(buffer.data[index + 1]);
    float b = static_cast<float>(buffer.data[index + 2]);
    return LUMA_RED_WEIGHT_BT601 * r + LUMA_GREEN_WEIGHT_BT601 * g + LUMA_BLUE_WEIGHT_BT601 * b;
  }
  return static_cast<float>(buffer.data[index]);
}

void GaussianBlurFilter::ApplyHorizontalBlur(const ImageBuffer& input, ImageBufferMut& output) {
  int radius = kernel_size_ / 2;
  Apply1DBlur<BlurDirection::HORIZONTAL>(input, output, kernel_, radius,
                                         static_cast<const IPixelGetter&>(*this));
}

void GaussianBlurFilter::ApplyVerticalBlur(const ImageBuffer& input, ImageBufferMut& output) {
  int radius = kernel_size_ / 2;
  Apply1DBlur<BlurDirection::VERTICAL>(input, output, kernel_, radius,
                                       static_cast<const IPixelGetter&>(*this));
}

void GaussianBlurFilter::ApplyFullBlur(const ImageBuffer& input, ImageBufferMut& output) {
  int radius = kernel_size_ / 2;

  for (int y = 0; y < input.height; y++) {
    for (int x = 0; x < input.width; x++) {
      float sum = Compute2DConvolution(input, x, y, kernel_, radius,
                                       static_cast<const IPixelGetter&>(*this));
      WritePixelToOutput(output, x, y, sum);
    }
  }
}

bool GaussianBlurFilter::Apply(FilterContext& context) {
  if (!separable_) {
    ApplyFullBlur(context.input, context.output);
    return true;
  }

  std::vector<unsigned char> temp_data(context.output.GetTotalSize());
  ImageBufferMut temp_mut(temp_data.data(), context.output.width, context.output.height,
                          context.output.channels);

  ApplyHorizontalBlur(context.input, temp_mut);
  ImageBuffer temp_read_buffer(temp_data.data(), context.output.width, context.output.height,
                               context.output.channels);
  ApplyVerticalBlur(temp_read_buffer, context.output);
  return true;
}

}  // namespace jrb::infrastructure::cpu

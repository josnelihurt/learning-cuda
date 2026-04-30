#include "src/cpp_accelerator/adapters/compute/cpu/blur_filter.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "src/cpp_accelerator/domain/interfaces/i_pixel_getter.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::infrastructure::cpu {

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::FilterType;
using jrb::domain::interfaces::IFilter;
using jrb::domain::interfaces::ImageBuffer;
using jrb::domain::interfaces::ImageBufferMut;
using jrb::domain::interfaces::IPixelGetter;

constexpr float kSqrt2Pi = 2.506628274631000242F;

namespace {
struct GenerateGaussianKernelParams {
  int size;
  float sigma;
};

std::vector<float> GenerateGaussianKernel(GenerateGaussianKernelParams params) {
  if (params.size % 2 == 0) {
    params.size++;
  }

  int radius = params.size / 2;
  std::vector<float> kernel(params.size);

  float sum = 0.0F;
  for (int i = 0; i < params.size; i++) {
    float x = static_cast<float>(i - radius);
    float value =
        std::exp(-0.5F * (x * x) / (params.sigma * params.sigma)) / (params.sigma * kSqrt2Pi);
    kernel[i] = value;
    sum += value;
  }

  for (int i = 0; i < params.size; i++) {
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
      case BorderMode::kClamp:
        return clamp_border_;
      case BorderMode::kReflect:
        return reflect_border_;
      case BorderMode::kWrap:
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

namespace {
enum class BlurDirection { kHorizontal, kVertical };
struct SampledPixelChannel {
  int x;
  int y;
  int channel_idx;
};
float GetPixelChannelValue(const ImageBuffer& buffer, const internal::IBorderClamper& clamper,
                           SampledPixelChannel sampled_pixel_channel) {
  int clamped_x = clamper.ClampX(sampled_pixel_channel.x, buffer.width);
  int clamped_y = clamper.ClampY(sampled_pixel_channel.y, buffer.height);

  int index = (clamped_y * buffer.width + clamped_x) * buffer.channels;
  if (sampled_pixel_channel.channel_idx < buffer.channels) {
    return static_cast<float>(buffer.data[index + sampled_pixel_channel.channel_idx]);
  }
  return 0.0F;
}

template <BlurDirection Direction>
void Apply1DBlur(const ImageBuffer& input, ImageBufferMut& output, const std::vector<float>& kernel,
                 int radius, const internal::IBorderClamper& clamper) {
  for (int y = 0; y < input.height; y++) {
    for (int x = 0; x < input.width; x++) {
      int output_idx = (y * output.width + x) * output.channels;

      for (int c = 0; c < output.channels; c++) {
        float sum = 0.0F;
        for (int k = -radius; k <= radius; k++) {
          int pixel_x = (Direction == BlurDirection::kHorizontal) ? x + k : x;
          int pixel_y = (Direction == BlurDirection::kHorizontal) ? y : y + k;
          float weight = kernel[k + radius];
          sum +=
              GetPixelChannelValue(input, clamper, {.x = pixel_x, .y = pixel_y, .channel_idx = c}) *
              weight;
        }
        output.data[output_idx + c] = static_cast<unsigned char>(std::clamp(sum, 0.0F, 255.0F));
      }
    }
  }
}

}  // anonymous namespace

GaussianBlurFilter::GaussianBlurFilter(int kernel_size, float sigma, BorderMode border_mode,
                                       bool separable)
    : kernel_size_(kernel_size), sigma_(sigma), border_mode_(border_mode), separable_(separable) {
  kernel_ = GenerateGaussianKernel({.size = kernel_size, .sigma = sigma});
  border_clamper_factory_ = std::make_unique<internal::BorderClamperFactory>();
}

GaussianBlurFilter::~GaussianBlurFilter() = default;

void GaussianBlurFilter::SetKernelSize(int size) {
  kernel_size_ = size;
  kernel_ = GenerateGaussianKernel({.size = size, .sigma = sigma_});
}

void GaussianBlurFilter::SetSigma(float sigma) {
  sigma_ = sigma;
  kernel_ = GenerateGaussianKernel({.size = kernel_size_, .sigma = sigma});
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
  return GetPixelChannelValue(buffer, clamper, {.x = x, .y = y, .channel_idx = 0});
}

void GaussianBlurFilter::ApplyHorizontalBlur(const ImageBuffer& input, ImageBufferMut& output) {
  int radius = kernel_size_ / 2;
  const internal::IBorderClamper& clamper = border_clamper_factory_->Get(border_mode_);
  Apply1DBlur<BlurDirection::kHorizontal>(input, output, kernel_, radius, clamper);
}

void GaussianBlurFilter::ApplyVerticalBlur(const ImageBuffer& input, ImageBufferMut& output) {
  int radius = kernel_size_ / 2;
  const internal::IBorderClamper& clamper = border_clamper_factory_->Get(border_mode_);
  Apply1DBlur<BlurDirection::kVertical>(input, output, kernel_, radius, clamper);
}

void GaussianBlurFilter::ApplyFullBlur(const ImageBuffer& input, ImageBufferMut& output) {
  int radius = kernel_size_ / 2;
  const internal::IBorderClamper& clamper = border_clamper_factory_->Get(border_mode_);

  for (int y = 0; y < input.height; y++) {
    for (int x = 0; x < input.width; x++) {
      int output_idx = (y * output.width + x) * output.channels;

      for (int c = 0; c < output.channels; c++) {
        float sum = 0.0F;
        for (int ky = -radius; ky <= radius; ky++) {
          for (int kx = -radius; kx <= radius; kx++) {
            float weight_y = kernel_[ky + radius];
            float weight_x = kernel_[kx + radius];
            float weight = weight_x * weight_y;
            sum +=
                GetPixelChannelValue(input, clamper, {.x = x + kx, .y = y + ky, .channel_idx = c}) *
                weight;
          }
        }
        output.data[output_idx + c] = static_cast<unsigned char>(std::clamp(sum, 0.0F, 255.0F));
      }
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

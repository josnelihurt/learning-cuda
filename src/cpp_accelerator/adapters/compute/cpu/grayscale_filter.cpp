#include "src/cpp_accelerator/adapters/compute/cpu/grayscale_filter.h"

#include <algorithm>

#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::infrastructure::cpu {

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::FilterType;
using jrb::domain::interfaces::ImageBuffer;
using jrb::domain::interfaces::ImageBufferMut;

namespace {
constexpr float kLumaRedWeightBt601 = 0.299F;
constexpr float kLumaGreenWeightBt601 = 0.587F;
constexpr float kLumaBlueWeightBt601 = 0.114F;

constexpr float kLumaRedWeightBt709 = 0.2126F;
constexpr float kLumaGreenWeightBt709 = 0.7152F;
constexpr float kLumaBlueWeightBt709 = 0.0722F;

constexpr float kLumaRedWeightLuminosity = 0.21F;
constexpr float kLumaGreenWeightLuminosity = 0.72F;
constexpr float kLumaBlueWeightLuminosity = 0.07F;
}  // namespace

GrayscaleFilter::GrayscaleFilter(GrayscaleAlgorithm algorithm) : algorithm_(algorithm) {}

void GrayscaleFilter::SetAlgorithm(GrayscaleAlgorithm algorithm) {
  algorithm_ = algorithm;
}

GrayscaleAlgorithm GrayscaleFilter::GetAlgorithm() const {
  return algorithm_;
}

FilterType GrayscaleFilter::GetType() const {
  return FilterType::GRAYSCALE;
}

bool GrayscaleFilter::IsInPlace() const {
  return false;
}

unsigned char GrayscaleFilter::CalculateGrayscaleValue(unsigned char r, unsigned char g,
                                                       unsigned char b) const {
  switch (algorithm_) {
    case GrayscaleAlgorithm::BT601:
      return static_cast<unsigned char>(kLumaRedWeightBt601 * static_cast<float>(r) +
                                        kLumaGreenWeightBt601 * static_cast<float>(g) +
                                        kLumaBlueWeightBt601 * static_cast<float>(b));
    case GrayscaleAlgorithm::BT709:
      return static_cast<unsigned char>(kLumaRedWeightBt709 * static_cast<float>(r) +
                                        kLumaGreenWeightBt709 * static_cast<float>(g) +
                                        kLumaBlueWeightBt709 * static_cast<float>(b));
    case GrayscaleAlgorithm::Average:
      return static_cast<unsigned char>(
          (static_cast<int>(r) + static_cast<int>(g) + static_cast<int>(b)) / 3);
    case GrayscaleAlgorithm::Lightness: {
      unsigned char max_val = std::max({r, g, b});
      unsigned char min_val = std::min({r, g, b});
      return static_cast<unsigned char>((static_cast<int>(max_val) + static_cast<int>(min_val)) /
                                        2);
    }
    case GrayscaleAlgorithm::Luminosity:
      return static_cast<unsigned char>(kLumaRedWeightLuminosity * static_cast<float>(r) +
                                        kLumaGreenWeightLuminosity * static_cast<float>(g) +
                                        kLumaBlueWeightLuminosity * static_cast<float>(b));
    default:
      return static_cast<unsigned char>(kLumaRedWeightBt601 * static_cast<float>(r) +
                                        kLumaGreenWeightBt601 * static_cast<float>(g) +
                                        kLumaBlueWeightBt601 * static_cast<float>(b));
  }
}

bool GrayscaleFilter::Apply(FilterContext& context) {
  if (!context.input.IsValid() || !context.output.IsValid()) {
    return false;
  }

  int total_pixels = context.input.width * context.input.height;
  int input_channels = context.input.channels;
  int output_channels = 1;

  for (int i = 0; i < total_pixels; i++) {
    int input_idx = i * input_channels;
    int output_idx = i * output_channels;

    if (input_channels >= 3) {
      unsigned char r = context.input.data[input_idx];
      unsigned char g = context.input.data[input_idx + 1];
      unsigned char b = context.input.data[input_idx + 2];
      context.output.data[output_idx] = CalculateGrayscaleValue(r, g, b);
    } else {
      context.output.data[output_idx] = context.input.data[input_idx];
    }
  }

  return true;
}

}  // namespace jrb::infrastructure::cpu

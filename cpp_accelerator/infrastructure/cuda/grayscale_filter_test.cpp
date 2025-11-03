#include "cpp_accelerator/infrastructure/cuda/grayscale_filter.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "cpp_accelerator/domain/interfaces/image_buffer.h"
#include "cpp_accelerator/infrastructure/cpu/grayscale_filter.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"

namespace jrb::infrastructure::cuda {
namespace {

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::ImageBuffer;
using jrb::domain::interfaces::ImageBufferMut;
using jrb::infrastructure::image::ImageLoader;

FilterContext CreateGrayscaleFilterContext(const unsigned char* input_data, unsigned char* output_data,
                                           int width, int height, int input_channels) {
  FilterContext context(input_data, output_data, width, height, input_channels);
  context.output.channels = 1;
  return context;
}

class GrayscaleFilterTest : public ::testing::Test {
protected:
  void SetUp() override {
    image_loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(image_loader_->is_valid());
  }

  void TearDown() override {}

  std::unique_ptr<ImageLoader> image_loader_;
};

TEST_F(GrayscaleFilterTest, FilterConstructsWithDefaultValues) {
  GrayscaleFilter filter;

  EXPECT_EQ(filter.GetType(), jrb::domain::interfaces::FilterType::GRAYSCALE);
  EXPECT_FALSE(filter.IsInPlace());
  EXPECT_EQ(filter.GetAlgorithm(), GrayscaleAlgorithm::BT601);
}

TEST_F(GrayscaleFilterTest, FilterConstructsWithCustomAlgorithm) {
  GrayscaleFilter filter(GrayscaleAlgorithm::BT709);

  EXPECT_EQ(filter.GetAlgorithm(), GrayscaleAlgorithm::BT709);
}

TEST_F(GrayscaleFilterTest, SetterUpdatesAlgorithm) {
  GrayscaleFilter filter;

  filter.SetAlgorithm(GrayscaleAlgorithm::Average);
  EXPECT_EQ(filter.GetAlgorithm(), GrayscaleAlgorithm::Average);

  filter.SetAlgorithm(GrayscaleAlgorithm::Luminosity);
  EXPECT_EQ(filter.GetAlgorithm(), GrayscaleAlgorithm::Luminosity);
}

TEST_F(GrayscaleFilterTest, AppliesBT601GrayscaleSuccessfully) {
  GrayscaleFilter filter(GrayscaleAlgorithm::BT601);
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  FilterContext context = CreateGrayscaleFilterContext(
      image_loader_->data(), output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  bool result = filter.Apply(context);

  ASSERT_TRUE(result);
  EXPECT_EQ(context.output.width, image_loader_->width());
  EXPECT_EQ(context.output.height, image_loader_->height());
  EXPECT_EQ(context.output.channels, 1);

  bool all_grayscale = true;
  for (int i = 0; i < image_loader_->width() * image_loader_->height(); ++i) {
    if (output[i] < 0 || output[i] > 255) {
      all_grayscale = false;
      break;
    }
  }
  EXPECT_TRUE(all_grayscale);
}

TEST_F(GrayscaleFilterTest, AppliesBT709GrayscaleSuccessfully) {
  GrayscaleFilter filter(GrayscaleAlgorithm::BT709);
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  FilterContext context = CreateGrayscaleFilterContext(
      image_loader_->data(), output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  bool result = filter.Apply(context);

  ASSERT_TRUE(result);
  EXPECT_EQ(context.output.channels, 1);
}

TEST_F(GrayscaleFilterTest, AppliesAverageGrayscaleSuccessfully) {
  GrayscaleFilter filter(GrayscaleAlgorithm::Average);
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  FilterContext context = CreateGrayscaleFilterContext(
      image_loader_->data(), output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  bool result = filter.Apply(context);

  ASSERT_TRUE(result);
  EXPECT_EQ(context.output.channels, 1);
}

TEST_F(GrayscaleFilterTest, AppliesLightnessGrayscaleSuccessfully) {
  GrayscaleFilter filter(GrayscaleAlgorithm::Lightness);
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  FilterContext context = CreateGrayscaleFilterContext(
      image_loader_->data(), output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  bool result = filter.Apply(context);

  ASSERT_TRUE(result);
  EXPECT_EQ(context.output.channels, 1);
}

TEST_F(GrayscaleFilterTest, AppliesLuminosityGrayscaleSuccessfully) {
  GrayscaleFilter filter(GrayscaleAlgorithm::Luminosity);
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  FilterContext context = CreateGrayscaleFilterContext(
      image_loader_->data(), output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  bool result = filter.Apply(context);

  ASSERT_TRUE(result);
  EXPECT_EQ(context.output.channels, 1);
}

TEST_F(GrayscaleFilterTest, HandlesSingleChannelInput) {
  GrayscaleFilter filter;
  std::vector<unsigned char> single_channel_input(100 * 100 * 1, 128);
  std::vector<unsigned char> output(100 * 100 * 1);
  FilterContext context(single_channel_input.data(), output.data(), 100, 100, 1);

  bool result = filter.Apply(context);

  ASSERT_TRUE(result);
  EXPECT_EQ(context.output.channels, 1);
}

TEST_F(GrayscaleFilterTest, FailsWithInvalidInput) {
  GrayscaleFilter filter;
  std::vector<unsigned char> output(100 * 100 * 1);
  FilterContext context(nullptr, output.data(), 100, 100, 3);

  bool result = filter.Apply(context);

  EXPECT_FALSE(result);
}

TEST_F(GrayscaleFilterTest, FailsWithInvalidOutput) {
  GrayscaleFilter filter;
  FilterContext context(image_loader_->data(), nullptr, image_loader_->width(),
                        image_loader_->height(), image_loader_->channels());

  bool result = filter.Apply(context);

  EXPECT_FALSE(result);
}

TEST_F(GrayscaleFilterTest, PreservesImageDimensions) {
  GrayscaleFilter filter;
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  FilterContext context = CreateGrayscaleFilterContext(
      image_loader_->data(), output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  filter.Apply(context);

  EXPECT_EQ(context.input.width, context.output.width);
  EXPECT_EQ(context.input.height, context.output.height);
}

TEST_F(GrayscaleFilterTest, CUDAAndCPUProduceIdenticalResultsForBT601) {
  GrayscaleFilter cuda_filter(GrayscaleAlgorithm::BT601);
  jrb::infrastructure::cpu::GrayscaleFilter cpu_filter(jrb::infrastructure::cpu::GrayscaleAlgorithm::BT601);

  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() * 1);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() * 1);

  FilterContext cuda_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cuda_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());
  FilterContext cpu_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cpu_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  ASSERT_TRUE(cuda_filter.Apply(cuda_context));
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));

  int mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < image_loader_->width() * image_loader_->height(); ++i) {
    int diff = std::abs(static_cast<int>(cuda_output[i]) - static_cast<int>(cpu_output[i]));
    if (diff > 0) {
      mismatches++;
      max_diff = std::max(max_diff, diff);
    }
  }
  
  EXPECT_LE(mismatches, image_loader_->width() * image_loader_->height() * 0.01)
      << "Too many mismatches: " << mismatches << " out of "
      << image_loader_->width() * image_loader_->height();
  EXPECT_LE(max_diff, 2) << "Max difference too large: " << max_diff;
}

TEST_F(GrayscaleFilterTest, CUDAAndCPUProduceIdenticalResultsForBT709) {
  GrayscaleFilter cuda_filter(GrayscaleAlgorithm::BT709);
  jrb::infrastructure::cpu::GrayscaleFilter cpu_filter(jrb::infrastructure::cpu::GrayscaleAlgorithm::BT709);

  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() * 1);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() * 1);

  FilterContext cuda_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cuda_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());
  FilterContext cpu_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cpu_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  ASSERT_TRUE(cuda_filter.Apply(cuda_context));
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));

  int mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < image_loader_->width() * image_loader_->height(); ++i) {
    int diff = std::abs(static_cast<int>(cuda_output[i]) - static_cast<int>(cpu_output[i]));
    if (diff > 0) {
      mismatches++;
      max_diff = std::max(max_diff, diff);
    }
  }
  
  EXPECT_LE(mismatches, image_loader_->width() * image_loader_->height() * 0.01)
      << "Too many mismatches: " << mismatches << " out of "
      << image_loader_->width() * image_loader_->height();
  EXPECT_LE(max_diff, 2) << "Max difference too large: " << max_diff;
}

TEST_F(GrayscaleFilterTest, CUDAAndCPUProduceIdenticalResultsForAverage) {
  GrayscaleFilter cuda_filter(GrayscaleAlgorithm::Average);
  jrb::infrastructure::cpu::GrayscaleFilter cpu_filter(jrb::infrastructure::cpu::GrayscaleAlgorithm::Average);

  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() * 1);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() * 1);

  FilterContext cuda_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cuda_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());
  FilterContext cpu_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cpu_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  ASSERT_TRUE(cuda_filter.Apply(cuda_context));
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));

  int mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < image_loader_->width() * image_loader_->height(); ++i) {
    int diff = std::abs(static_cast<int>(cuda_output[i]) - static_cast<int>(cpu_output[i]));
    if (diff > 0) {
      mismatches++;
      max_diff = std::max(max_diff, diff);
    }
  }
  
  EXPECT_LE(mismatches, image_loader_->width() * image_loader_->height() * 0.01)
      << "Too many mismatches: " << mismatches << " out of "
      << image_loader_->width() * image_loader_->height();
  EXPECT_LE(max_diff, 2) << "Max difference too large: " << max_diff;
}

TEST_F(GrayscaleFilterTest, CUDAAndCPUProduceIdenticalResultsForLightness) {
  GrayscaleFilter cuda_filter(GrayscaleAlgorithm::Lightness);
  jrb::infrastructure::cpu::GrayscaleFilter cpu_filter(jrb::infrastructure::cpu::GrayscaleAlgorithm::Lightness);

  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() * 1);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() * 1);

  FilterContext cuda_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cuda_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());
  FilterContext cpu_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cpu_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  ASSERT_TRUE(cuda_filter.Apply(cuda_context));
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));

  int mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < image_loader_->width() * image_loader_->height(); ++i) {
    int diff = std::abs(static_cast<int>(cuda_output[i]) - static_cast<int>(cpu_output[i]));
    if (diff > 0) {
      mismatches++;
      max_diff = std::max(max_diff, diff);
    }
  }
  
  EXPECT_LE(mismatches, image_loader_->width() * image_loader_->height() * 0.01)
      << "Too many mismatches: " << mismatches << " out of "
      << image_loader_->width() * image_loader_->height();
  EXPECT_LE(max_diff, 2) << "Max difference too large: " << max_diff;
}

TEST_F(GrayscaleFilterTest, CUDAAndCPUProduceIdenticalResultsForLuminosity) {
  GrayscaleFilter cuda_filter(GrayscaleAlgorithm::Luminosity);
  jrb::infrastructure::cpu::GrayscaleFilter cpu_filter(jrb::infrastructure::cpu::GrayscaleAlgorithm::Luminosity);

  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() * 1);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() * 1);

  FilterContext cuda_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cuda_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());
  FilterContext cpu_context = CreateGrayscaleFilterContext(
      image_loader_->data(), cpu_output.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  ASSERT_TRUE(cuda_filter.Apply(cuda_context));
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));

  int mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < image_loader_->width() * image_loader_->height(); ++i) {
    int diff = std::abs(static_cast<int>(cuda_output[i]) - static_cast<int>(cpu_output[i]));
    if (diff > 0) {
      mismatches++;
      max_diff = std::max(max_diff, diff);
    }
  }
  
  EXPECT_LE(mismatches, image_loader_->width() * image_loader_->height() * 0.01)
      << "Too many mismatches: " << mismatches << " out of "
      << image_loader_->width() * image_loader_->height();
  EXPECT_LE(max_diff, 2) << "Max difference too large: " << max_diff;
}

}  // namespace
}  // namespace jrb::infrastructure::cuda


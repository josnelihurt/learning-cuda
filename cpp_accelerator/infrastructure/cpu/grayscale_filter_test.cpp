#include "cpp_accelerator/infrastructure/cpu/grayscale_filter.h"

#include <gtest/gtest.h>
#include <vector>

#include "cpp_accelerator/domain/interfaces/image_buffer.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"

namespace jrb::infrastructure::cpu {
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

TEST_F(GrayscaleFilterTest, DifferentAlgorithmsProduceDifferentResults) {
  std::vector<unsigned char> output_bt601(image_loader_->width() * image_loader_->height() * 1);
  std::vector<unsigned char> output_bt709(image_loader_->width() * image_loader_->height() * 1);
  std::vector<unsigned char> output_average(image_loader_->width() * image_loader_->height() * 1);

  GrayscaleFilter filter_bt601(GrayscaleAlgorithm::BT601);
  GrayscaleFilter filter_bt709(GrayscaleAlgorithm::BT709);
  GrayscaleFilter filter_average(GrayscaleAlgorithm::Average);

  FilterContext context_bt601 = CreateGrayscaleFilterContext(
      image_loader_->data(), output_bt601.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());
  FilterContext context_bt709 = CreateGrayscaleFilterContext(
      image_loader_->data(), output_bt709.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());
  FilterContext context_average = CreateGrayscaleFilterContext(
      image_loader_->data(), output_average.data(), image_loader_->width(), image_loader_->height(),
      image_loader_->channels());

  ASSERT_TRUE(filter_bt601.Apply(context_bt601));
  ASSERT_TRUE(filter_bt709.Apply(context_bt709));
  ASSERT_TRUE(filter_average.Apply(context_average));

  bool has_differences = false;
  for (int i = 0; i < image_loader_->width() * image_loader_->height(); ++i) {
    if (output_bt601[i] != output_bt709[i] || output_bt601[i] != output_average[i]) {
      has_differences = true;
      break;
    }
  }

  EXPECT_TRUE(has_differences);
}

}  // namespace
}  // namespace jrb::infrastructure::cpu


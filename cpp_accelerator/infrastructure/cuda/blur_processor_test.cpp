#include "cpp_accelerator/infrastructure/cuda/blur_processor.h"

#include <gtest/gtest.h>
#include <vector>

#include "cpp_accelerator/domain/interfaces/image_buffer.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"

namespace jrb::infrastructure::cuda {
namespace {

using jrb::domain::interfaces::FilterContext;
using jrb::infrastructure::image::ImageLoader;

class CudaGaussianBlurFilterTest : public ::testing::Test {
protected:
  void SetUp() override {
    image_loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(image_loader_->is_valid());
  }

  void TearDown() override {}

  std::unique_ptr<ImageLoader> image_loader_;
};

TEST_F(CudaGaussianBlurFilterTest, FilterConstructsWithDefaultValues) {
  // Arrange & Act
  CudaGaussianBlurFilter filter;

  // Assert
  EXPECT_EQ(filter.GetType(), jrb::domain::interfaces::FilterType::BLUR);
  EXPECT_FALSE(filter.IsInPlace());
  EXPECT_EQ(filter.GetKernelSize(), 5);
  EXPECT_FLOAT_EQ(filter.GetSigma(), 1.0F);
  EXPECT_EQ(filter.GetBorderMode(), BorderMode::REFLECT);
}

TEST_F(CudaGaussianBlurFilterTest, FilterConstructsWithCustomValues) {
  // Arrange & Act
  CudaGaussianBlurFilter filter(7, 2.5F, BorderMode::CLAMP, true);

  // Assert
  EXPECT_EQ(filter.GetKernelSize(), 7);
  EXPECT_FLOAT_EQ(filter.GetSigma(), 2.5F);
  EXPECT_EQ(filter.GetBorderMode(), BorderMode::CLAMP);
}

TEST_F(CudaGaussianBlurFilterTest, SettersUpdateFilterConfiguration) {
  // Arrange
  CudaGaussianBlurFilter filter;

  // Act & Assert
  filter.SetKernelSize(9);
  EXPECT_EQ(filter.GetKernelSize(), 9);

  filter.SetSigma(3.0F);
  EXPECT_FLOAT_EQ(filter.GetSigma(), 3.0F);

  filter.SetBorderMode(BorderMode::WRAP);
  EXPECT_EQ(filter.GetBorderMode(), BorderMode::WRAP);
}

TEST_F(CudaGaussianBlurFilterTest, AppliesBlurToImageSuccessfully) {
  // Arrange
  CudaGaussianBlurFilter filter(5, 1.0F, BorderMode::REFLECT, true);
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() *
                                    image_loader_->channels());
  FilterContext context(image_loader_->data(), output.data(), image_loader_->width(),
                        image_loader_->height(), image_loader_->channels());
  ASSERT_TRUE(context.input.IsValid());
  ASSERT_TRUE(context.output.IsValid());

  // Act
  bool result = filter.Apply(context);

  // Assert
  ASSERT_TRUE(result);
  bool has_variation = false;
  for (size_t i = 0; i < output.size() && !has_variation; ++i) {
    if (image_loader_->data()[i] != output[i]) {
      has_variation = true;
    }
  }
  EXPECT_TRUE(has_variation);
}

TEST_F(CudaGaussianBlurFilterTest, BlurPreservesImageDimensions) {
  // Arrange
  CudaGaussianBlurFilter filter;
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() *
                                    image_loader_->channels());
  FilterContext context(image_loader_->data(), output.data(), image_loader_->width(),
                        image_loader_->height(), image_loader_->channels());

  // Act
  filter.Apply(context);

  // Assert
  EXPECT_TRUE(context.input.IsValid());
  EXPECT_TRUE(context.output.IsValid());
  EXPECT_EQ(context.input.width, context.output.width);
  EXPECT_EQ(context.input.height, context.output.height);
  EXPECT_EQ(context.input.channels, context.output.channels);
  EXPECT_EQ(context.input.width, image_loader_->width());
  EXPECT_EQ(context.input.height, image_loader_->height());
  EXPECT_EQ(context.input.channels, image_loader_->channels());
}

TEST_F(CudaGaussianBlurFilterTest, DifferentSigmaValuesProduceDifferentResults) {
  // Arrange
  int kernel_size = 5;
  BorderMode border_mode = BorderMode::REFLECT;
  CudaGaussianBlurFilter filter1(kernel_size, 0.5F, border_mode, true);
  CudaGaussianBlurFilter filter2(kernel_size, 2.0F, border_mode, true);
  std::vector<unsigned char> output1(image_loader_->width() * image_loader_->height() *
                                     image_loader_->channels());
  std::vector<unsigned char> output2(image_loader_->width() * image_loader_->height() *
                                     image_loader_->channels());
  FilterContext context1(image_loader_->data(), output1.data(), image_loader_->width(),
                         image_loader_->height(), image_loader_->channels());
  FilterContext context2(image_loader_->data(), output2.data(), image_loader_->width(),
                         image_loader_->height(), image_loader_->channels());
  ASSERT_TRUE(context1.input.IsValid() && context2.input.IsValid());

  // Act
  ASSERT_TRUE(filter1.Apply(context1));
  ASSERT_TRUE(filter2.Apply(context2));

  // Assert
  bool results_differ = false;
  for (size_t i = 0; i < output1.size() && !results_differ; ++i) {
    if (output1[i] != output2[i]) {
      results_differ = true;
    }
  }
  EXPECT_TRUE(results_differ);
}

TEST_F(CudaGaussianBlurFilterTest, DifferentBorderModesProduceDifferentResults) {
  // Arrange
  int kernel_size = 5;
  float sigma = 1.0F;
  CudaGaussianBlurFilter filter1(kernel_size, sigma, BorderMode::CLAMP, true);
  CudaGaussianBlurFilter filter2(kernel_size, sigma, BorderMode::REFLECT, true);
  CudaGaussianBlurFilter filter3(kernel_size, sigma, BorderMode::WRAP, true);
  std::vector<unsigned char> output1(image_loader_->width() * image_loader_->height() *
                                     image_loader_->channels());
  std::vector<unsigned char> output2(image_loader_->width() * image_loader_->height() *
                                     image_loader_->channels());
  std::vector<unsigned char> output3(image_loader_->width() * image_loader_->height() *
                                     image_loader_->channels());
  FilterContext context1(image_loader_->data(), output1.data(), image_loader_->width(),
                         image_loader_->height(), image_loader_->channels());
  FilterContext context2(image_loader_->data(), output2.data(), image_loader_->width(),
                         image_loader_->height(), image_loader_->channels());
  FilterContext context3(image_loader_->data(), output3.data(), image_loader_->width(),
                         image_loader_->height(), image_loader_->channels());
  ASSERT_TRUE(context1.input.IsValid() && context2.input.IsValid() && context3.input.IsValid());

  // Act
  ASSERT_TRUE(filter1.Apply(context1));
  ASSERT_TRUE(filter2.Apply(context2));
  ASSERT_TRUE(filter3.Apply(context3));

  // Assert
  bool modes_produce_different_results = false;
  for (size_t i = 0; i < output1.size() && !modes_produce_different_results; ++i) {
    if (output1[i] != output2[i] || output2[i] != output3[i]) {
      modes_produce_different_results = true;
    }
  }
  EXPECT_TRUE(modes_produce_different_results);
}

TEST_F(CudaGaussianBlurFilterTest, AppliesCorrectlyToSmallImage) {
  // Arrange
  constexpr int test_width = 10;
  constexpr int test_height = 10;
  constexpr int test_channels = 3;
  std::vector<unsigned char> test_input(test_width * test_height * test_channels, 128);
  std::vector<unsigned char> test_output(test_width * test_height * test_channels);
  FilterContext context(test_input.data(), test_output.data(), test_width, test_height,
                        test_channels);
  CudaGaussianBlurFilter filter(3, 1.0F, BorderMode::REFLECT, true);
  ASSERT_TRUE(context.input.IsValid());

  // Act
  bool result = filter.Apply(context);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_TRUE(context.output.IsValid());
}

TEST_F(CudaGaussianBlurFilterTest, SingleChannelImageProcessesCorrectly) {
  // Arrange
  constexpr int test_width = 50;
  constexpr int test_height = 50;
  constexpr int test_channels = 1;
  std::vector<unsigned char> test_input(test_width * test_height * test_channels, 100);
  std::vector<unsigned char> test_output(test_width * test_height * test_channels);
  FilterContext context(test_input.data(), test_output.data(), test_width, test_height,
                        test_channels);
  CudaGaussianBlurFilter filter(5, 1.0F, BorderMode::REFLECT, true);
  ASSERT_TRUE(context.input.IsValid());

  // Act
  bool result = filter.Apply(context);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_TRUE(context.output.IsValid());
}

TEST_F(CudaGaussianBlurFilterTest, LargeKernelSizeProducesHeavyBlur) {
  // Arrange
  CudaGaussianBlurFilter filter(15, 1.0F, BorderMode::REFLECT, true);
  CudaGaussianBlurFilter reference_filter(5, 1.0F, BorderMode::REFLECT, true);
  std::vector<unsigned char> output_large(image_loader_->width() * image_loader_->height() *
                                          image_loader_->channels());
  std::vector<unsigned char> output_reference(image_loader_->width() * image_loader_->height() *
                                              image_loader_->channels());
  FilterContext context_large(image_loader_->data(), output_large.data(), image_loader_->width(),
                              image_loader_->height(), image_loader_->channels());
  FilterContext context_reference(image_loader_->data(), output_reference.data(),
                                  image_loader_->width(), image_loader_->height(),
                                  image_loader_->channels());
  ASSERT_TRUE(context_large.input.IsValid() && context_reference.input.IsValid());

  // Act
  ASSERT_TRUE(filter.Apply(context_large));
  ASSERT_TRUE(reference_filter.Apply(context_reference));

  // Assert
  int differences_count = 0;
  for (size_t i = 0; i < output_large.size(); ++i) {
    if (output_large[i] != output_reference[i]) {
      differences_count++;
    }
  }
  EXPECT_GT(differences_count, 0);
}

TEST_F(CudaGaussianBlurFilterTest, AllBorderModesHandleEdgePixels) {
  // Arrange
  std::vector<BorderMode> modes = {BorderMode::CLAMP, BorderMode::REFLECT, BorderMode::WRAP};
  std::vector<std::vector<unsigned char>> outputs(
      modes.size(), std::vector<unsigned char>(image_loader_->width() * image_loader_->height() *
                                               image_loader_->channels()));
  std::vector<FilterContext> contexts;
  for (size_t i = 0; i < modes.size(); ++i) {
    contexts.emplace_back(image_loader_->data(), outputs[i].data(), image_loader_->width(),
                          image_loader_->height(), image_loader_->channels());
    ASSERT_TRUE(contexts[i].input.IsValid());
  }

  // Act
  for (size_t i = 0; i < modes.size(); ++i) {
    CudaGaussianBlurFilter filter(5, 1.0F, modes[i], true);
    ASSERT_TRUE(filter.Apply(contexts[i]));
  }

  // Assert
  for (const auto& context : contexts) {
    EXPECT_TRUE(context.output.IsValid());
  }
}

}  // namespace
}  // namespace jrb::infrastructure::cuda

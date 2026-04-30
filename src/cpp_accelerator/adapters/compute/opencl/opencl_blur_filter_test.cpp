#include "src/cpp_accelerator/adapters/compute/opencl/opencl_blur_filter.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "src/cpp_accelerator/adapters/compute/opencl/opencl_context.h"
#include "src/cpp_accelerator/adapters/image_io/image_loader.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::infrastructure::opencl {
namespace {

using jrb::domain::interfaces::FilterContext;
using jrb::infrastructure::image::ImageLoader;

bool OpenCLAvailable() {
  return OpenCLContext::GetInstance().available();
}

class OpenCLBlurFilterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(loader_->is_valid());
  }

  std::unique_ptr<ImageLoader> loader_;
};

// --- Construction ---

TEST_F(OpenCLBlurFilterTest, Success_ConstructsWithCorrectType) {
  // Arrange & Act
  OpenCLBlurFilter sut;

  // Assert
  EXPECT_EQ(sut.GetType(), jrb::domain::interfaces::FilterType::BLUR);
}

TEST_F(OpenCLBlurFilterTest, Success_IsNotInPlace) {
  // Arrange & Act
  OpenCLBlurFilter sut;

  // Assert
  EXPECT_FALSE(sut.IsInPlace());
}

// --- Apply: basic ---

TEST_F(OpenCLBlurFilterTest, Success_ApplyReturnsTrueWhenContextAvailable) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange
  OpenCLBlurFilter sut;
  const int w = loader_->width(), h = loader_->height(), ch = loader_->channels();
  std::vector<unsigned char> output(w * h * ch);
  FilterContext ctx(loader_->data(), output.data(), w, h, ch);

  // Act
  bool result = sut.Apply(ctx);

  // Assert
  EXPECT_TRUE(result);
}

TEST_F(OpenCLBlurFilterTest, Success_OutputDimensionsPreserved) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange
  OpenCLBlurFilter sut;
  const int w = loader_->width(), h = loader_->height(), ch = loader_->channels();
  std::vector<unsigned char> output(w * h * ch);
  FilterContext ctx(loader_->data(), output.data(), w, h, ch);

  // Act
  ASSERT_TRUE(sut.Apply(ctx));

  // Assert
  EXPECT_EQ(ctx.input.width, w);
  EXPECT_EQ(ctx.input.height, h);
  EXPECT_EQ(ctx.input.channels, ch);
  EXPECT_EQ(ctx.output.width, w);
  EXPECT_EQ(ctx.output.height, h);
  EXPECT_EQ(ctx.output.channels, ch);
}

TEST_F(OpenCLBlurFilterTest, Success_BlurActuallyChangesPixels) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange: use lena.png which has natural variation
  OpenCLBlurFilter sut;
  const int w = loader_->width(), h = loader_->height(), ch = loader_->channels();
  std::vector<unsigned char> output(w * h * ch);
  FilterContext ctx(loader_->data(), output.data(), w, h, ch);

  // Act
  ASSERT_TRUE(sut.Apply(ctx));

  // Assert: blur must change at least some pixels vs the original
  bool has_change = false;
  for (int i = 0; i < w * h * ch && !has_change; ++i) {
    if (output[i] != loader_->data()[i]) has_change = true;
  }
  EXPECT_TRUE(has_change);
}

// --- Apply: constant image ---

TEST_F(OpenCLBlurFilterTest, Success_ConstantImageRemainsConstantAfterBlur) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange: flat image — blur of a constant is a constant
  constexpr int w = 16, h = 16, ch = 3;
  constexpr unsigned char kVal = 128;
  std::vector<unsigned char> input(w * h * ch, kVal);
  std::vector<unsigned char> output(w * h * ch, 0);
  OpenCLBlurFilter sut;
  FilterContext ctx(input.data(), output.data(), w, h, ch);

  // Act
  ASSERT_TRUE(sut.Apply(ctx));

  // Assert: every output pixel should equal kVal (Gaussian of constant = constant)
  for (int i = 0; i < w * h * ch; ++i) {
    EXPECT_EQ(output[i], kVal) << "mismatch at index " << i;
  }
}

// --- Apply: single-channel image ---

TEST_F(OpenCLBlurFilterTest, Success_SingleChannelImageProcessedCorrectly) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange
  constexpr int w = 32, h = 32, ch = 1;
  std::vector<unsigned char> input(w * h * ch, 100);
  std::vector<unsigned char> output(w * h * ch, 0);
  OpenCLBlurFilter sut;
  FilterContext ctx(input.data(), output.data(), w, h, ch);

  // Act
  bool result = sut.Apply(ctx);

  // Assert
  EXPECT_TRUE(result);
  EXPECT_TRUE(ctx.output.IsValid());
}

// --- Apply: blur produces a smoother image ---

TEST_F(OpenCLBlurFilterTest, Success_BlurReducesHighFrequencyVariation) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange: checkerboard pattern maximises high-frequency content
  const int w = 64, h = 64, ch = 3;
  std::vector<unsigned char> input(w * h * ch);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      unsigned char val = ((x + y) % 2 == 0) ? 255 : 0;
      for (int c = 0; c < ch; ++c) input[(y * w + x) * ch + c] = val;
    }
  }
  std::vector<unsigned char> output(w * h * ch, 0);
  OpenCLBlurFilter sut;
  FilterContext ctx(input.data(), output.data(), w, h, ch);

  // Act
  ASSERT_TRUE(sut.Apply(ctx));

  // Assert: sum of absolute differences between neighbours is lower in output
  long input_variation = 0, output_variation = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x + 1 < w; ++x) {
      for (int c = 0; c < ch; ++c) {
        input_variation +=
            std::abs(static_cast<int>(input[(y * w + x) * ch + c]) -
                     static_cast<int>(input[(y * w + x + 1) * ch + c]));
        output_variation +=
            std::abs(static_cast<int>(output[(y * w + x) * ch + c]) -
                     static_cast<int>(output[(y * w + x + 1) * ch + c]));
      }
    }
  }
  EXPECT_LT(output_variation, input_variation)
      << "Blur should reduce high-frequency variation";
}


TEST_F(OpenCLBlurFilterTest, Success_SmallImageHandledCorrectly) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange: 3×3 RGB image (smaller than the 5-tap kernel radius)
  constexpr int w = 3, h = 3, ch = 3;
  std::vector<unsigned char> input(w * h * ch, 64);
  std::vector<unsigned char> output(w * h * ch, 0);
  OpenCLBlurFilter sut;
  FilterContext ctx(input.data(), output.data(), w, h, ch);

  // Act
  bool result = sut.Apply(ctx);

  // Assert: must complete without error; constant input → constant output
  ASSERT_TRUE(result);
  for (int i = 0; i < w * h * ch; ++i) {
    EXPECT_EQ(output[i], 64) << "index " << i;
  }
}

}  // namespace
}  // namespace jrb::infrastructure::opencl

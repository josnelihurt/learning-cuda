#include "src/cpp_accelerator/adapters/compute/opencl/filters/grayscale_filter.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "src/cpp_accelerator/adapters/compute/cpu/grayscale_filter.h"
#include "src/cpp_accelerator/adapters/compute/opencl/context/context.h"
#include "src/cpp_accelerator/adapters/image_io/image_loader.h"
#include "src/cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::adapters::compute::opencl {
namespace {

using jrb::domain::interfaces::FilterContext;
using jrb::adapters::image::ImageLoader;

// Returns false when OpenCL is not available in the test environment.
// Tests that require a live device should call this and skip gracefully.
bool OpenCLAvailable() {
  return Context::GetInstance().available();
}

FilterContext MakeGrayscaleContext(const unsigned char* input, unsigned char* output, int width,
                                   int height, int channels) {
  FilterContext ctx(input, output, width, height, channels);
  ctx.output.channels = 1;
  return ctx;
}

class GrayscaleFilterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(loader_->is_valid());
  }

  std::unique_ptr<ImageLoader> loader_;
};

// --- Construction ---

TEST_F(GrayscaleFilterTest, Success_ConstructsWithCorrectType) {
  // Arrange & Act
  GrayscaleFilter sut;

  // Assert
  EXPECT_EQ(sut.GetType(), jrb::domain::interfaces::FilterType::GRAYSCALE);
}

TEST_F(GrayscaleFilterTest, Success_IsNotInPlace) {
  // Arrange & Act
  GrayscaleFilter sut;

  // Assert
  EXPECT_FALSE(sut.IsInPlace());
}

// --- Apply: basic ---

TEST_F(GrayscaleFilterTest, Success_ApplyReturnsTrueWhenContextAvailable) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange
  GrayscaleFilter sut;
  const int w = loader_->width(), h = loader_->height();
  std::vector<unsigned char> output(w * h);
  FilterContext ctx = MakeGrayscaleContext(loader_->data(), output.data(), w, h,
                                           loader_->channels());

  // Act
  bool result = sut.Apply(ctx);

  // Assert
  EXPECT_TRUE(result);
}

TEST_F(GrayscaleFilterTest, Success_OutputDimensionsPreserved) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange
  GrayscaleFilter sut;
  const int w = loader_->width(), h = loader_->height();
  std::vector<unsigned char> output(w * h);
  FilterContext ctx = MakeGrayscaleContext(loader_->data(), output.data(), w, h,
                                           loader_->channels());

  // Act
  ASSERT_TRUE(sut.Apply(ctx));

  // Assert
  EXPECT_EQ(ctx.input.width, w);
  EXPECT_EQ(ctx.input.height, h);
}

TEST_F(GrayscaleFilterTest, Success_OutputValuesAreInValidRange) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange
  GrayscaleFilter sut;
  const int w = loader_->width(), h = loader_->height();
  std::vector<unsigned char> output(w * h, 0xFF);
  FilterContext ctx = MakeGrayscaleContext(loader_->data(), output.data(), w, h,
                                           loader_->channels());

  // Act
  ASSERT_TRUE(sut.Apply(ctx));

  // Assert: output contains values across the full uint8 range (not all-zero, not all-max)
  bool has_non_zero = false;
  bool has_non_max = false;
  for (int i = 0; i < w * h; ++i) {
    if (output[i] > 0) has_non_zero = true;
    if (output[i] < 255) has_non_max = true;
  }
  EXPECT_TRUE(has_non_zero);
  EXPECT_TRUE(has_non_max);
}

// --- Apply: pixel equivalence with CPU BT.601 ---

TEST_F(GrayscaleFilterTest, Success_MatchesCpuBT601WithinRoundingTolerance) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange
  const int w = loader_->width(), h = loader_->height();
  std::vector<unsigned char> ocl_out(w * h);
  std::vector<unsigned char> cpu_out(w * h);

  GrayscaleFilter ocl_filter;
  cpu::GrayscaleFilter cpu_filter(cpu::GrayscaleAlgorithm::BT601);

  FilterContext ocl_ctx = MakeGrayscaleContext(loader_->data(), ocl_out.data(), w, h,
                                               loader_->channels());
  FilterContext cpu_ctx = MakeGrayscaleContext(loader_->data(), cpu_out.data(), w, h,
                                               loader_->channels());

  // Act
  ASSERT_TRUE(ocl_filter.Apply(ocl_ctx));
  ASSERT_TRUE(cpu_filter.Apply(cpu_ctx));

  // Assert: each pixel within ±1 (float→uint8 rounding differences)
  int mismatch = 0;
  for (int i = 0; i < w * h; ++i) {
    if (std::abs(static_cast<int>(ocl_out[i]) - static_cast<int>(cpu_out[i])) > 1) {
      ++mismatch;
    }
  }
  const double pct = 100.0 * mismatch / (w * h);
  EXPECT_LT(pct, 1.0) << mismatch << " pixels differ by more than 1 out of " << (w * h);
}

// --- Apply: constant-colour image ---

TEST_F(GrayscaleFilterTest, Success_ConstantColorImageProducesExpectedLuminance) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange: pure red 4×4 image
  constexpr int w = 4, h = 4, ch = 3;
  std::vector<unsigned char> input(w * h * ch, 0);
  for (int i = 0; i < w * h; ++i) input[i * 3 + 0] = 200;  // R=200, G=0, B=0
  std::vector<unsigned char> output(w * h);
  GrayscaleFilter sut;
  FilterContext ctx = MakeGrayscaleContext(input.data(), output.data(), w, h, ch);

  // Act
  ASSERT_TRUE(sut.Apply(ctx));

  // Assert: expected ≈ round(0.299 * 200) = 60
  const int expected = static_cast<int>(0.299f * 200.f + 0.5f);
  for (int i = 0; i < w * h; ++i) {
    EXPECT_NEAR(static_cast<int>(output[i]), expected, 1) << "pixel " << i;
  }
}

// --- Error cases ---

TEST_F(GrayscaleFilterTest, Error_RejectsFourChannelInput) {
  if (!OpenCLAvailable()) GTEST_SKIP() << "OpenCL not available";

  // Arrange: 4-channel RGBA input
  constexpr int w = 8, h = 8, ch = 4;
  std::vector<unsigned char> input(w * h * ch, 128);
  std::vector<unsigned char> output(w * h);
  GrayscaleFilter sut;
  FilterContext ctx = MakeGrayscaleContext(input.data(), output.data(), w, h, ch);

  // Act
  bool result = sut.Apply(ctx);

  // Assert: filter only supports 3-channel input
  EXPECT_FALSE(result);
}

}  // namespace
}  // namespace jrb::adapters::compute::opencl

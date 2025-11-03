#include "cpp_accelerator/infrastructure/cpu/blur_filter.h"
#include "cpp_accelerator/infrastructure/cuda/blur_processor.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "cpp_accelerator/domain/interfaces/image_buffer.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"

namespace jrb::infrastructure {
namespace {

using jrb::domain::interfaces::FilterContext;
using jrb::infrastructure::image::ImageLoader;

class BlurEquivalenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    image_loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(image_loader_->is_valid());
  }

  void TearDown() override {}

  std::unique_ptr<ImageLoader> image_loader_;
};

TEST_F(BlurEquivalenceTest, CpuAndCudaProduceSameResultsWithDefaultSettings) {
  // Arrange
  cpu::GaussianBlurFilter cpu_filter;
  cuda::CudaGaussianBlurFilter cuda_filter;
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() *
                                        image_loader_->channels());
  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() *
                                         image_loader_->channels());
  FilterContext cpu_context(image_loader_->data(), cpu_output.data(), image_loader_->width(),
                            image_loader_->height(), image_loader_->channels());
  FilterContext cuda_context(image_loader_->data(), cuda_output.data(), image_loader_->width(),
                             image_loader_->height(), image_loader_->channels());
  ASSERT_TRUE(cpu_context.input.IsValid() && cuda_context.input.IsValid());

  // Act
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));
  ASSERT_TRUE(cuda_filter.Apply(cuda_context));

  // Assert
  int pixels_different = 0;
  const int tolerance = 2;
  int max_diff = 0;
  for (size_t i = 0; i < cpu_output.size(); ++i) {
    int diff = std::abs(static_cast<int>(cpu_output[i]) - static_cast<int>(cuda_output[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > tolerance) {
      pixels_different++;
    }
  }
  double percent_different = (static_cast<double>(pixels_different) / cpu_output.size()) * 100.0;
  EXPECT_LT(percent_different, 10.0) << "Max difference: " << max_diff;
}

TEST_F(BlurEquivalenceTest, CpuAndCudaProduceSameResultsWithCustomSigma) {
  // Arrange
  cpu::GaussianBlurFilter cpu_filter(5, 2.0F, cpu::BorderMode::REFLECT, true);
  cuda::CudaGaussianBlurFilter cuda_filter(5, 2.0F, cuda::BorderMode::REFLECT, true);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() *
                                        image_loader_->channels());
  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() *
                                         image_loader_->channels());
  FilterContext cpu_context(image_loader_->data(), cpu_output.data(), image_loader_->width(),
                            image_loader_->height(), image_loader_->channels());
  FilterContext cuda_context(image_loader_->data(), cuda_output.data(), image_loader_->width(),
                             image_loader_->height(), image_loader_->channels());
  ASSERT_TRUE(cpu_context.input.IsValid() && cuda_context.input.IsValid());

  // Act
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));
  ASSERT_TRUE(cuda_filter.Apply(cuda_context));

  // Assert
  int pixels_different = 0;
  const int tolerance = 2;
  int max_diff = 0;
  for (size_t i = 0; i < cpu_output.size(); ++i) {
    int diff = std::abs(static_cast<int>(cpu_output[i]) - static_cast<int>(cuda_output[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > tolerance) {
      pixels_different++;
    }
  }
  double percent_different = (static_cast<double>(pixels_different) / cpu_output.size()) * 100.0;
  EXPECT_LT(percent_different, 10.0) << "Max difference: " << max_diff;
}

TEST_F(BlurEquivalenceTest, CpuAndCudaProduceSameResultsWithClampBorder) {
  // Arrange
  cpu::GaussianBlurFilter cpu_filter(5, 1.0F, cpu::BorderMode::CLAMP, true);
  cuda::CudaGaussianBlurFilter cuda_filter(5, 1.0F, cuda::BorderMode::CLAMP, true);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() *
                                        image_loader_->channels());
  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() *
                                         image_loader_->channels());
  FilterContext cpu_context(image_loader_->data(), cpu_output.data(), image_loader_->width(),
                            image_loader_->height(), image_loader_->channels());
  FilterContext cuda_context(image_loader_->data(), cuda_output.data(), image_loader_->width(),
                             image_loader_->height(), image_loader_->channels());
  ASSERT_TRUE(cpu_context.input.IsValid() && cuda_context.input.IsValid());

  // Act
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));
  ASSERT_TRUE(cuda_filter.Apply(cuda_context));

  // Assert
  int pixels_different = 0;
  const int tolerance = 2;
  int max_diff = 0;
  for (size_t i = 0; i < cpu_output.size(); ++i) {
    int diff = std::abs(static_cast<int>(cpu_output[i]) - static_cast<int>(cuda_output[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > tolerance) {
      pixels_different++;
    }
  }
  double percent_different = (static_cast<double>(pixels_different) / cpu_output.size()) * 100.0;
  EXPECT_LT(percent_different, 10.0) << "Max difference: " << max_diff;
}

TEST_F(BlurEquivalenceTest, CpuAndCudaProduceSameResultsWithWrapBorder) {
  // Arrange
  cpu::GaussianBlurFilter cpu_filter(5, 1.0F, cpu::BorderMode::WRAP, true);
  cuda::CudaGaussianBlurFilter cuda_filter(5, 1.0F, cuda::BorderMode::WRAP, true);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() *
                                        image_loader_->channels());
  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() *
                                         image_loader_->channels());
  FilterContext cpu_context(image_loader_->data(), cpu_output.data(), image_loader_->width(),
                            image_loader_->height(), image_loader_->channels());
  FilterContext cuda_context(image_loader_->data(), cuda_output.data(), image_loader_->width(),
                             image_loader_->height(), image_loader_->channels());
  ASSERT_TRUE(cpu_context.input.IsValid() && cuda_context.input.IsValid());

  // Act
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));
  ASSERT_TRUE(cuda_filter.Apply(cuda_context));

  // Assert
  int pixels_different = 0;
  const int tolerance = 2;
  int max_diff = 0;
  for (size_t i = 0; i < cpu_output.size(); ++i) {
    int diff = std::abs(static_cast<int>(cpu_output[i]) - static_cast<int>(cuda_output[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > tolerance) {
      pixels_different++;
    }
  }
  double percent_different = (static_cast<double>(pixels_different) / cpu_output.size()) * 100.0;
  EXPECT_LT(percent_different, 10.0) << "Max difference: " << max_diff;
}

TEST_F(BlurEquivalenceTest, CpuAndCudaProduceSameResultsWithLargeKernel) {
  // Arrange
  cpu::GaussianBlurFilter cpu_filter(15, 1.0F, cpu::BorderMode::REFLECT, true);
  cuda::CudaGaussianBlurFilter cuda_filter(15, 1.0F, cuda::BorderMode::REFLECT, true);
  std::vector<unsigned char> cpu_output(image_loader_->width() * image_loader_->height() *
                                        image_loader_->channels());
  std::vector<unsigned char> cuda_output(image_loader_->width() * image_loader_->height() *
                                         image_loader_->channels());
  FilterContext cpu_context(image_loader_->data(), cpu_output.data(), image_loader_->width(),
                            image_loader_->height(), image_loader_->channels());
  FilterContext cuda_context(image_loader_->data(), cuda_output.data(), image_loader_->width(),
                             image_loader_->height(), image_loader_->channels());
  ASSERT_TRUE(cpu_context.input.IsValid() && cuda_context.input.IsValid());

  // Act
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));
  ASSERT_TRUE(cuda_filter.Apply(cuda_context));

  // Assert
  int pixels_different = 0;
  const int tolerance = 2;
  int max_diff = 0;
  for (size_t i = 0; i < cpu_output.size(); ++i) {
    int diff = std::abs(static_cast<int>(cpu_output[i]) - static_cast<int>(cuda_output[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > tolerance) {
      pixels_different++;
    }
  }
  double percent_different = (static_cast<double>(pixels_different) / cpu_output.size()) * 100.0;
  EXPECT_LT(percent_different, 10.0) << "Max difference: " << max_diff;
}

TEST_F(BlurEquivalenceTest, CpuAndCudaProduceSameResultsWithSmallImage) {
  // Arrange
  constexpr int test_width = 50;
  constexpr int test_height = 50;
  constexpr int test_channels = 3;
  std::vector<unsigned char> test_input(test_width * test_height * test_channels, 128);
  std::vector<unsigned char> cpu_output(test_width * test_height * test_channels);
  std::vector<unsigned char> cuda_output(test_width * test_height * test_channels);
  FilterContext cpu_context(test_input.data(), cpu_output.data(), test_width, test_height,
                            test_channels);
  FilterContext cuda_context(test_input.data(), cuda_output.data(), test_width, test_height,
                             test_channels);
  cpu::GaussianBlurFilter cpu_filter(5, 1.0F, cpu::BorderMode::REFLECT, true);
  cuda::CudaGaussianBlurFilter cuda_filter(5, 1.0F, cuda::BorderMode::REFLECT, true);
  ASSERT_TRUE(cpu_context.input.IsValid() && cuda_context.input.IsValid());

  // Act
  ASSERT_TRUE(cpu_filter.Apply(cpu_context));
  ASSERT_TRUE(cuda_filter.Apply(cuda_context));

  // Assert
  int pixels_different = 0;
  const int tolerance = 2;
  int max_diff = 0;
  for (size_t i = 0; i < cpu_output.size(); ++i) {
    int diff = std::abs(static_cast<int>(cpu_output[i]) - static_cast<int>(cuda_output[i]));
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (diff > tolerance) {
      pixels_different++;
    }
  }
  double percent_different = (static_cast<double>(pixels_different) / cpu_output.size()) * 100.0;
  EXPECT_LT(percent_different, 10.0) << "Max difference: " << max_diff;
}

}  // namespace
}  // namespace jrb::infrastructure

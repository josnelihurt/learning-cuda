#include "cpp_accelerator/application/pipeline/filter_pipeline.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "cpp_accelerator/application/pipeline/buffer_pool.h"
#include "cpp_accelerator/domain/interfaces/image_buffer.h"
#include "cpp_accelerator/infrastructure/cpu/blur_filter.h"
#include "cpp_accelerator/infrastructure/cpu/grayscale_filter.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"

namespace jrb::application::pipeline {
namespace {

using jrb::domain::interfaces::FilterType;
using jrb::domain::interfaces::ImageBuffer;
using jrb::domain::interfaces::ImageBufferMut;
using jrb::infrastructure::cpu::BorderMode;
using jrb::infrastructure::cpu::GaussianBlurFilter;
using jrb::infrastructure::cpu::GrayscaleAlgorithm;
using jrb::infrastructure::cpu::GrayscaleFilter;
using jrb::infrastructure::image::ImageLoader;

class FilterPipelineTest : public ::testing::Test {
protected:
  void SetUp() override {
    image_loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(image_loader_->is_valid());
  }

  void TearDown() override {}

  std::unique_ptr<ImageLoader> image_loader_;
};

TEST_F(FilterPipelineTest, SingleFilterPipelineAppliesSuccessfully) {
  // Arrange
  FilterPipeline pipeline;
  auto grayscale = std::make_unique<GrayscaleFilter>(GrayscaleAlgorithm::BT601);
  pipeline.AddFilter(std::move(grayscale));

  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  ImageBuffer input_buffer(image_loader_->data(), image_loader_->width(), image_loader_->height(),
                           image_loader_->channels());
  ImageBufferMut output_buffer(output.data(), image_loader_->width(), image_loader_->height(), 1);

  // Act
  bool result = pipeline.Apply(input_buffer, output_buffer);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_TRUE(output_buffer.IsValid());
  EXPECT_EQ(output_buffer.width, image_loader_->width());
  EXPECT_EQ(output_buffer.height, image_loader_->height());
  EXPECT_EQ(output_buffer.channels, 1);
}

TEST_F(FilterPipelineTest, MultipleFiltersApplyInSequence) {
  // Arrange
  FilterPipeline pipeline;
  auto grayscale = std::make_unique<GrayscaleFilter>(GrayscaleAlgorithm::BT601);
  auto blur = std::make_unique<GaussianBlurFilter>(5, 1.0F, BorderMode::REFLECT, true);
  pipeline.AddFilter(std::move(grayscale));
  pipeline.AddFilter(std::move(blur));

  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  ImageBuffer input_buffer(image_loader_->data(), image_loader_->width(), image_loader_->height(),
                           image_loader_->channels());
  ImageBufferMut output_buffer(output.data(), image_loader_->width(), image_loader_->height(), 1);

  // Act
  bool result = pipeline.Apply(input_buffer, output_buffer);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_TRUE(output_buffer.IsValid());
  EXPECT_EQ(output_buffer.width, image_loader_->width());
  EXPECT_EQ(output_buffer.height, image_loader_->height());
  EXPECT_EQ(output_buffer.channels, 1);
}

TEST_F(FilterPipelineTest, EmptyPipelineReturnsFalse) {
  // Arrange
  FilterPipeline pipeline;
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() *
                                    image_loader_->channels());
  ImageBuffer input_buffer(image_loader_->data(), image_loader_->width(), image_loader_->height(),
                           image_loader_->channels());
  ImageBufferMut output_buffer(output.data(), image_loader_->width(), image_loader_->height(),
                               image_loader_->channels());

  // Act
  bool result = pipeline.Apply(input_buffer, output_buffer);

  // Assert
  EXPECT_FALSE(result);
}

TEST_F(FilterPipelineTest, PipelinePreservesImageDimensions) {
  // Arrange
  FilterPipeline pipeline;
  auto blur = std::make_unique<GaussianBlurFilter>(5, 1.0F, BorderMode::REFLECT, true);
  pipeline.AddFilter(std::move(blur));

  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() *
                                    image_loader_->channels());
  ImageBuffer input_buffer(image_loader_->data(), image_loader_->width(), image_loader_->height(),
                           image_loader_->channels());
  ImageBufferMut output_buffer(output.data(), image_loader_->width(), image_loader_->height(),
                               image_loader_->channels());

  // Act
  bool result = pipeline.Apply(input_buffer, output_buffer);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_EQ(output_buffer.width, input_buffer.width);
  EXPECT_EQ(output_buffer.height, input_buffer.height);
  EXPECT_EQ(output_buffer.channels, input_buffer.channels);
}

TEST_F(FilterPipelineTest, BufferPoolReusesBuffers) {
  // Arrange
  auto buffer_pool = std::make_unique<BufferPool>(2);
  FilterPipeline pipeline(std::move(buffer_pool));
  auto grayscale = std::make_unique<GrayscaleFilter>(GrayscaleAlgorithm::BT601);
  auto blur = std::make_unique<GaussianBlurFilter>(5, 1.0F, BorderMode::REFLECT, true);
  pipeline.AddFilter(std::move(grayscale));
  pipeline.AddFilter(std::move(blur));

  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  ImageBuffer input_buffer(image_loader_->data(), image_loader_->width(), image_loader_->height(),
                           image_loader_->channels());
  ImageBufferMut output_buffer(output.data(), image_loader_->width(), image_loader_->height(), 1);

  // Act
  bool result = pipeline.Apply(input_buffer, output_buffer);

  // Assert
  ASSERT_TRUE(result);
}

TEST_F(FilterPipelineTest, GetFilterCountReturnsCorrectNumber) {
  // Arrange
  FilterPipeline pipeline;

  // Act & Assert
  EXPECT_EQ(pipeline.GetFilterCount(), 0);

  pipeline.AddFilter(std::make_unique<GrayscaleFilter>());
  EXPECT_EQ(pipeline.GetFilterCount(), 1);

  pipeline.AddFilter(std::make_unique<GaussianBlurFilter>());
  EXPECT_EQ(pipeline.GetFilterCount(), 2);
}

TEST_F(FilterPipelineTest, ClearRemovesAllFilters) {
  // Arrange
  FilterPipeline pipeline;
  pipeline.AddFilter(std::make_unique<GrayscaleFilter>());
  pipeline.AddFilter(std::make_unique<GaussianBlurFilter>());
  ASSERT_EQ(pipeline.GetFilterCount(), 2);

  // Act
  pipeline.Clear();

  // Assert
  EXPECT_EQ(pipeline.GetFilterCount(), 0);
}

TEST_F(FilterPipelineTest, PipelineWithThreeFiltersAppliesSuccessfully) {
  // Arrange
  FilterPipeline pipeline;
  auto grayscale1 = std::make_unique<GrayscaleFilter>(GrayscaleAlgorithm::BT601);
  auto blur = std::make_unique<GaussianBlurFilter>(5, 1.0F, BorderMode::REFLECT, true);
  auto grayscale2 = std::make_unique<GrayscaleFilter>(GrayscaleAlgorithm::BT709);
  pipeline.AddFilter(std::move(grayscale1));
  pipeline.AddFilter(std::move(blur));
  pipeline.AddFilter(std::move(grayscale2));

  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  ImageBuffer input_buffer(image_loader_->data(), image_loader_->width(), image_loader_->height(),
                           image_loader_->channels());
  ImageBufferMut output_buffer(output.data(), image_loader_->width(), image_loader_->height(), 1);

  // Act
  bool result = pipeline.Apply(input_buffer, output_buffer);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_TRUE(output_buffer.IsValid());
}

}  // namespace
}  // namespace jrb::application::pipeline

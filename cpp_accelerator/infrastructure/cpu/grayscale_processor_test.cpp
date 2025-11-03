#include "cpp_accelerator/infrastructure/cpu/grayscale_processor.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "cpp_accelerator/domain/interfaces/image_sink.h"
#include "cpp_accelerator/domain/interfaces/image_source.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"

namespace jrb::infrastructure::cpu {
namespace {

using jrb::domain::interfaces::IImageSink;
using jrb::infrastructure::image::ImageLoader;

class TestImageSink : public IImageSink {
public:
  bool write(const char* filepath, const unsigned char* data, int width, int height,
             int channels) override {
    width_ = width;
    height_ = height;
    channels_ = channels;
    data_.assign(data, data + (width * height * channels));
    write_called_ = true;
    return true;
  }

  int GetWidth() const { return width_; }
  int GetHeight() const { return height_; }
  int GetChannels() const { return channels_; }
  const std::vector<unsigned char>& GetData() const { return data_; }
  bool WasWriteCalled() const { return write_called_; }

private:
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;
  std::vector<unsigned char> data_;
  bool write_called_ = false;
};

class GrayscaleProcessorTest : public ::testing::Test {
protected:
  void SetUp() override {
    image_loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(image_loader_->is_valid());
    processor_ = std::make_unique<CpuGrayscaleProcessor>();
    sink_ = std::make_unique<TestImageSink>();
  }

  void TearDown() override {}

  std::unique_ptr<ImageLoader> image_loader_;
  std::unique_ptr<CpuGrayscaleProcessor> processor_;
  std::unique_ptr<TestImageSink> sink_;
};

TEST_F(GrayscaleProcessorTest, ConstructsWithDefaultAlgorithm) {
  // Arrange & Act
  CpuGrayscaleProcessor processor;

  // Assert
  EXPECT_EQ(processor.get_algorithm(), GrayscaleAlgorithm::BT601);
}

TEST_F(GrayscaleProcessorTest, ConstructsWithCustomAlgorithm) {
  // Arrange & Act
  CpuGrayscaleProcessor processor(GrayscaleAlgorithm::BT709);

  // Assert
  EXPECT_EQ(processor.get_algorithm(), GrayscaleAlgorithm::BT709);
}

TEST_F(GrayscaleProcessorTest, SetAlgorithmUpdatesAlgorithm) {
  // Arrange
  CpuGrayscaleProcessor processor;

  // Act
  processor.set_algorithm(GrayscaleAlgorithm::Average);

  // Assert
  EXPECT_EQ(processor.get_algorithm(), GrayscaleAlgorithm::Average);
}

TEST_F(GrayscaleProcessorTest, ProcessConvertsImageToGrayscale) {
  // Arrange
  ASSERT_TRUE(image_loader_->is_valid());

  // Act
  bool result = processor_->process(*image_loader_, *sink_, "");

  // Assert
  ASSERT_TRUE(result);
  ASSERT_TRUE(sink_->WasWriteCalled());
  EXPECT_EQ(sink_->GetWidth(), image_loader_->width());
  EXPECT_EQ(sink_->GetHeight(), image_loader_->height());
  EXPECT_EQ(sink_->GetChannels(), 1);
}

TEST_F(GrayscaleProcessorTest, ProcessPreservesImageDimensions) {
  // Arrange
  ASSERT_TRUE(image_loader_->is_valid());

  // Act
  processor_->process(*image_loader_, *sink_, "");

  // Assert
  EXPECT_EQ(sink_->GetWidth(), image_loader_->width());
  EXPECT_EQ(sink_->GetHeight(), image_loader_->height());
}

TEST_F(GrayscaleProcessorTest, DifferentAlgorithmsProduceDifferentResults) {
  // Arrange
  CpuGrayscaleProcessor processor1(GrayscaleAlgorithm::BT601);
  CpuGrayscaleProcessor processor2(GrayscaleAlgorithm::BT709);
  TestImageSink sink1, sink2;

  // Act
  bool result1 = processor1.process(*image_loader_, sink1, "");
  bool result2 = processor2.process(*image_loader_, sink2, "");

  // Assert
  ASSERT_TRUE(result1);
  ASSERT_TRUE(result2);

  const auto& data1 = sink1.GetData();
  const auto& data2 = sink2.GetData();
  ASSERT_EQ(data1.size(), data2.size());

  bool results_differ = false;
  for (size_t i = 0; i < data1.size() && !results_differ; ++i) {
    if (data1[i] != data2[i]) {
      results_differ = true;
    }
  }
  EXPECT_TRUE(results_differ);
}

TEST_F(GrayscaleProcessorTest, ProcessWithAverageAlgorithm) {
  // Arrange
  CpuGrayscaleProcessor processor(GrayscaleAlgorithm::Average);
  TestImageSink sink;

  // Act
  bool result = processor.process(*image_loader_, sink, "");

  // Assert
  ASSERT_TRUE(result);
  ASSERT_TRUE(sink.WasWriteCalled());
  EXPECT_EQ(sink.GetChannels(), 1);
}

TEST_F(GrayscaleProcessorTest, ProcessWithLightnessAlgorithm) {
  // Arrange
  CpuGrayscaleProcessor processor(GrayscaleAlgorithm::Lightness);
  TestImageSink sink;

  // Act
  bool result = processor.process(*image_loader_, sink, "");

  // Assert
  ASSERT_TRUE(result);
  ASSERT_TRUE(sink.WasWriteCalled());
  EXPECT_EQ(sink.GetChannels(), 1);
}

TEST_F(GrayscaleProcessorTest, ProcessWithLuminosityAlgorithm) {
  // Arrange
  CpuGrayscaleProcessor processor(GrayscaleAlgorithm::Luminosity);
  TestImageSink sink;

  // Act
  bool result = processor.process(*image_loader_, sink, "");

  // Assert
  ASSERT_TRUE(result);
  ASSERT_TRUE(sink.WasWriteCalled());
  EXPECT_EQ(sink.GetChannels(), 1);
}

TEST_F(GrayscaleProcessorTest, OutputDataIsValidGrayscale) {
  // Arrange
  ASSERT_TRUE(image_loader_->is_valid());

  // Act
  processor_->process(*image_loader_, *sink_, "");

  // Assert
  const auto& output_data = sink_->GetData();
  EXPECT_FALSE(output_data.empty());
  EXPECT_EQ(output_data.size(), static_cast<size_t>(image_loader_->width() * image_loader_->height()));

  for (unsigned char pixel : output_data) {
    EXPECT_LE(pixel, 255);
  }
}

}  // namespace
}  // namespace jrb::infrastructure::cpu


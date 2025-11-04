#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include "common.pb.h"
#include "image_processor_service.pb.h"
#pragma GCC diagnostic pop

#include "cpp_accelerator/domain/interfaces/image_buffer.h"
#include "cpp_accelerator/infrastructure/cpu/blur_filter.h"
#include "cpp_accelerator/infrastructure/image/image_loader.h"
#include "processor_api.h"

namespace jrb::ports::shared_lib {
namespace {

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::ImageBuffer;
using jrb::domain::interfaces::ImageBufferMut;
using jrb::infrastructure::cpu::BorderMode;
using jrb::infrastructure::cpu::GaussianBlurFilter;
using jrb::infrastructure::image::ImageLoader;

class BlurE2ETest : public ::testing::Test {
protected:
  void SetUp() override {
    image_loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(image_loader_->is_valid());
  }

  void TearDown() override {}

  std::unique_ptr<ImageLoader> image_loader_;

  bool InitializeLibrary() {
    cuda_learning::InitRequest init_req;
    init_req.set_cuda_device_id(0);

    std::string request_data = init_req.SerializeAsString();
    uint8_t* response_buf = nullptr;
    int response_len = 0;

    bool result =
        processor_init(reinterpret_cast<const uint8_t*>(request_data.data()),
                       static_cast<int>(request_data.size()), &response_buf, &response_len);

    if (response_buf) {
      processor_free_response(response_buf);
    }

    return result;
  }

  void CleanupLibrary() { processor_cleanup(); }

  bool ProcessImageGrayscale(const unsigned char* image_data, int width, int height, int channels,
                             std::vector<unsigned char>& output_data) {
    cuda_learning::ProcessImageRequest proc_req;
    proc_req.set_width(width);
    proc_req.set_height(height);
    proc_req.set_channels(channels);
    proc_req.set_image_data(image_data, static_cast<size_t>(width) * height * channels);
    proc_req.add_filters(cuda_learning::FILTER_TYPE_GRAYSCALE);
    proc_req.set_accelerator(cuda_learning::ACCELERATOR_TYPE_CPU);
    proc_req.set_grayscale_type(cuda_learning::GRAYSCALE_TYPE_BT601);

    std::string request_data = proc_req.SerializeAsString();
    uint8_t* response_buf = nullptr;
    int response_len = 0;

    bool result = processor_process_image(reinterpret_cast<const uint8_t*>(request_data.data()),
                                          static_cast<int>(request_data.size()), &response_buf,
                                          &response_len);

    if (!result || !response_buf) {
      if (response_buf) {
        processor_free_response(response_buf);
      }
      return false;
    }

    cuda_learning::ProcessImageResponse proc_resp;
    if (!proc_resp.ParseFromArray(response_buf, response_len)) {
      processor_free_response(response_buf);
      return false;
    }

    if (proc_resp.code() != 0) {
      processor_free_response(response_buf);
      return false;
    }

    output_data.assign(proc_resp.image_data().begin(), proc_resp.image_data().end());
    processor_free_response(response_buf);
    return true;
  }
};

TEST_F(BlurE2ETest, E2E_GrayscaleProcessingViaPublicAPI) {
  // Arrange
  ASSERT_TRUE(InitializeLibrary());

  // Act
  std::vector<unsigned char> output_data;
  bool result =
      ProcessImageGrayscale(image_loader_->data(), image_loader_->width(), image_loader_->height(),
                            image_loader_->channels(), output_data);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_FALSE(output_data.empty());
  EXPECT_EQ(output_data.size(),
            static_cast<size_t>(image_loader_->width() * image_loader_->height() * 1));

  CleanupLibrary();
}

TEST_F(BlurE2ETest, E2E_BlurFilterInternalValidation) {
  // Arrange
  ASSERT_TRUE(InitializeLibrary());

  std::vector<unsigned char> grayscale_output;
  ASSERT_TRUE(ProcessImageGrayscale(image_loader_->data(), image_loader_->width(),
                                    image_loader_->height(), image_loader_->channels(),
                                    grayscale_output));

  GaussianBlurFilter blur_filter(5, 1.0F, BorderMode::REFLECT, true);
  std::vector<unsigned char> blur_output(grayscale_output.size());

  ImageBuffer input_buffer(grayscale_output.data(), image_loader_->width(), image_loader_->height(),
                           1);
  ImageBufferMut output_buffer(blur_output.data(), image_loader_->width(), image_loader_->height(),
                               1);

  FilterContext context(grayscale_output.data(), blur_output.data(), image_loader_->width(),
                        image_loader_->height(), 1);

  // Act
  bool result = blur_filter.Apply(context);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_TRUE(context.output.IsValid());

  bool has_blur_effect = false;
  for (size_t i = 0; i < grayscale_output.size() && !has_blur_effect; ++i) {
    int diff = std::abs(static_cast<int>(grayscale_output[i]) - static_cast<int>(blur_output[i]));
    if (diff > 1) {
      has_blur_effect = true;
    }
  }
  EXPECT_TRUE(has_blur_effect);

  CleanupLibrary();
}

TEST_F(BlurE2ETest, E2E_CompletePipelineGrayscaleThenBlur) {
  // Arrange
  ASSERT_TRUE(InitializeLibrary());

  std::vector<unsigned char> grayscale_output;
  ASSERT_TRUE(ProcessImageGrayscale(image_loader_->data(), image_loader_->width(),
                                    image_loader_->height(), image_loader_->channels(),
                                    grayscale_output));

  GaussianBlurFilter blur_filter(7, 2.0F, BorderMode::REFLECT, true);
  std::vector<unsigned char> final_output(grayscale_output.size());

  FilterContext context(grayscale_output.data(), final_output.data(), image_loader_->width(),
                        image_loader_->height(), 1);

  // Act
  bool result = blur_filter.Apply(context);

  // Assert
  ASSERT_TRUE(result);
  EXPECT_TRUE(context.output.IsValid());
  EXPECT_EQ(context.output.width, image_loader_->width());
  EXPECT_EQ(context.output.height, image_loader_->height());
  EXPECT_EQ(context.output.channels, 1);

  for (unsigned char pixel : final_output) {
    EXPECT_LE(pixel, 255);
  }

  CleanupLibrary();
}

TEST_F(BlurE2ETest, E2E_LibraryVersionCheck) {
  // Arrange & Act
  processor_version_t version = processor_api_version();

  // Assert
  EXPECT_EQ(version.major, 2);
  EXPECT_EQ(version.minor, 1);
  EXPECT_EQ(version.patch, 0);
}

TEST_F(BlurE2ETest, E2E_MultipleGrayscaleCallsProduceConsistentResults) {
  // Arrange
  ASSERT_TRUE(InitializeLibrary());

  // Act
  std::vector<unsigned char> output1, output2;
  bool result1 = ProcessImageGrayscale(image_loader_->data(), image_loader_->width(),
                                       image_loader_->height(), image_loader_->channels(), output1);
  bool result2 = ProcessImageGrayscale(image_loader_->data(), image_loader_->width(),
                                       image_loader_->height(), image_loader_->channels(), output2);

  // Assert
  ASSERT_TRUE(result1);
  ASSERT_TRUE(result2);
  EXPECT_EQ(output1.size(), output2.size());

  bool outputs_match = true;
  for (size_t i = 0; i < output1.size() && outputs_match; ++i) {
    if (output1[i] != output2[i]) {
      outputs_match = false;
    }
  }
  EXPECT_TRUE(outputs_match);

  CleanupLibrary();
}

}  // namespace
}  // namespace jrb::ports::shared_lib

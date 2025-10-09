#include "infrastructure/image/image_writer.h"
#include <gtest/gtest.h>
#include <vector>
#include <fstream>
#include <filesystem>

namespace jrb::infrastructure::image {
namespace {

class ImageWriterTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Clean up any test files
        if (std::filesystem::exists(test_file_path_)) {
            std::filesystem::remove(test_file_path_);
        }
    }

    const char* test_file_path_ = "/tmp/test_image_write.png";
};

TEST_F(ImageWriterTest, WriteValidImage) {
    // Arrange
    ImageWriter uut;
    constexpr int width = 100;
    constexpr int height = 100;
    constexpr int channels = 3;
    std::vector<unsigned char> test_data(width * height * channels, 128);

    // Act
    bool result = uut.write(test_file_path_, test_data.data(), width, height, channels);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_TRUE(std::filesystem::exists(test_file_path_));
    
    // Verify file is not empty
    std::ifstream file(test_file_path_, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(file.is_open());
    auto file_size = file.tellg();
    EXPECT_GT(file_size, 0);
}

TEST_F(ImageWriterTest, WriteWithNullData) {
    // Arrange
    ImageWriter uut;
    constexpr int width = 100;
    constexpr int height = 100;
    constexpr int channels = 3;

    // Act
    bool result = uut.write(test_file_path_, nullptr, width, height, channels);

    // Assert
    EXPECT_FALSE(result);
    EXPECT_FALSE(std::filesystem::exists(test_file_path_));
}

TEST_F(ImageWriterTest, WriteWithZeroWidth) {
    // Arrange
    ImageWriter uut;
    constexpr int width = 0;
    constexpr int height = 100;
    constexpr int channels = 3;
    std::vector<unsigned char> test_data(100, 128);

    // Act
    bool result = uut.write(test_file_path_, test_data.data(), width, height, channels);

    // Assert
    EXPECT_FALSE(result);
    EXPECT_FALSE(std::filesystem::exists(test_file_path_));
}

TEST_F(ImageWriterTest, WriteWithZeroHeight) {
    // Arrange
    ImageWriter uut;
    constexpr int width = 100;
    constexpr int height = 0;
    constexpr int channels = 3;
    std::vector<unsigned char> test_data(100, 128);

    // Act
    bool result = uut.write(test_file_path_, test_data.data(), width, height, channels);

    // Assert
    EXPECT_FALSE(result);
    EXPECT_FALSE(std::filesystem::exists(test_file_path_));
}

TEST_F(ImageWriterTest, WriteWithZeroChannels) {
    // Arrange
    ImageWriter uut;
    constexpr int width = 100;
    constexpr int height = 100;
    constexpr int channels = 0;
    std::vector<unsigned char> test_data(width * height, 128);

    // Act
    bool result = uut.write(test_file_path_, test_data.data(), width, height, channels);

    // Assert
    EXPECT_FALSE(result);
    EXPECT_FALSE(std::filesystem::exists(test_file_path_));
}

TEST_F(ImageWriterTest, WriteWithNegativeDimensions) {
    // Arrange
    ImageWriter uut;
    constexpr int width = -100;
    constexpr int height = -100;
    constexpr int channels = -3;
    std::vector<unsigned char> test_data(100, 128);

    // Act
    bool result = uut.write(test_file_path_, test_data.data(), width, height, channels);

    // Assert
    EXPECT_FALSE(result);
    EXPECT_FALSE(std::filesystem::exists(test_file_path_));
}

TEST_F(ImageWriterTest, WriteSingleChannelImage) {
    // Arrange
    ImageWriter uut;
    constexpr int width = 50;
    constexpr int height = 50;
    constexpr int channels = 1;
    std::vector<unsigned char> test_data(width * height * channels, 200);

    // Act
    bool result = uut.write(test_file_path_, test_data.data(), width, height, channels);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_TRUE(std::filesystem::exists(test_file_path_));
}

TEST_F(ImageWriterTest, WriteFourChannelImage) {
    // Arrange
    ImageWriter uut;
    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int channels = 4;
    std::vector<unsigned char> test_data(width * height * channels, 255);

    // Act
    bool result = uut.write(test_file_path_, test_data.data(), width, height, channels);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_TRUE(std::filesystem::exists(test_file_path_));
}

TEST_F(ImageWriterTest, InterfaceImplementation) {
    // Arrange
    ImageWriter uut;
    domain::interfaces::IImageSink* interface_ptr = &uut;
    constexpr int width = 32;
    constexpr int height = 32;
    constexpr int channels = 3;
    std::vector<unsigned char> test_data(width * height * channels, 100);

    // Act
    bool result = interface_ptr->write(test_file_path_, test_data.data(), width, height, channels);

    // Assert
    EXPECT_TRUE(result);
    EXPECT_TRUE(std::filesystem::exists(test_file_path_));
}

}  // namespace
}  // namespace jrb::infrastructure::image

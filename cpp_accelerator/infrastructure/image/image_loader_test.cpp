#include "cpp_accelerator/infrastructure/image/image_loader.h"
#include <gtest/gtest.h>

namespace jrb::infrastructure::image {
namespace {

class ImageLoaderTest : public ::testing::Test {};

TEST_F(ImageLoaderTest, LoadDefaultImage) {
  // Arrange & Act
  ImageLoader uut;

  // Assert
  EXPECT_TRUE(uut.is_valid());
  EXPECT_TRUE(uut.is_loaded());
  EXPECT_GT(uut.width(), 0);
  EXPECT_GT(uut.height(), 0);
  EXPECT_GT(uut.channels(), 0);
  EXPECT_NE(uut.data(), nullptr);
}

TEST_F(ImageLoaderTest, LoadExistingImageByPath) {
  // Arrange
  const char* path = "data/static_images/lena.png";

  // Act
  ImageLoader uut(path);

  // Assert
  EXPECT_TRUE(uut.is_valid());
  EXPECT_TRUE(uut.is_loaded());
  EXPECT_GT(uut.width(), 0);
  EXPECT_GT(uut.height(), 0);
  EXPECT_GT(uut.channels(), 0);
  EXPECT_NE(uut.data(), nullptr);
}

TEST_F(ImageLoaderTest, LoadNonExistentImage) {
  // Arrange
  const char* path = "non_existent_file.png";

  // Act
  ImageLoader uut(path);

  // Assert
  EXPECT_FALSE(uut.is_valid());
  EXPECT_FALSE(uut.is_loaded());
  EXPECT_EQ(uut.width(), 0);
  EXPECT_EQ(uut.height(), 0);
  EXPECT_EQ(uut.channels(), 0);
  EXPECT_EQ(uut.data(), nullptr);
}

TEST_F(ImageLoaderTest, MoveConstructor) {
  // Arrange
  ImageLoader source("data/static_images/lena.png");
  ASSERT_TRUE(source.is_valid());
  int width = source.width();
  int height = source.height();
  int channels = source.channels();

  // Act
  ImageLoader uut(std::move(source));

  // Assert
  EXPECT_TRUE(uut.is_valid());
  EXPECT_EQ(uut.width(), width);
  EXPECT_EQ(uut.height(), height);
  EXPECT_EQ(uut.channels(), channels);
  EXPECT_NE(uut.data(), nullptr);

  // Source should be invalidated
  EXPECT_FALSE(source.is_valid());
  EXPECT_EQ(source.data(), nullptr);
}

TEST_F(ImageLoaderTest, MoveAssignment) {
  // Arrange
  ImageLoader source("data/static_images/lena.png");
  ASSERT_TRUE(source.is_valid());
  int width = source.width();
  int height = source.height();
  int channels = source.channels();

  ImageLoader uut;

  // Act
  uut = std::move(source);

  // Assert
  EXPECT_TRUE(uut.is_valid());
  EXPECT_EQ(uut.width(), width);
  EXPECT_EQ(uut.height(), height);
  EXPECT_EQ(uut.channels(), channels);
  EXPECT_NE(uut.data(), nullptr);

  // Source should be invalidated
  EXPECT_FALSE(source.is_valid());
  EXPECT_EQ(source.data(), nullptr);
}

TEST_F(ImageLoaderTest, InterfaceImplementation) {
  // Arrange
  ImageLoader uut("data/static_images/lena.png");
  domain::interfaces::IImageSource* interface_ptr = &uut;

  // Act & Assert
  EXPECT_TRUE(interface_ptr->is_valid());
  EXPECT_GT(interface_ptr->width(), 0);
  EXPECT_GT(interface_ptr->height(), 0);
  EXPECT_GT(interface_ptr->channels(), 0);
  EXPECT_NE(interface_ptr->data(), nullptr);
}

}  // namespace
}  // namespace jrb::infrastructure::image

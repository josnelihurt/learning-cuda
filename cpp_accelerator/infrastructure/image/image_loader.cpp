#include "cpp_accelerator/infrastructure/image/image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"

#include <spdlog/spdlog.h>
#include <cstring>

namespace jrb::infrastructure::image {

ImageLoader::ImageLoader() 
    : data_(nullptr), width_(0), height_(0), channels_(0) {
  data_ = stbi_load(kImagePath, &width_, &height_, &channels_, 0);
  
  if (data_ == nullptr) {
    spdlog::error("Failed to load image from {}", kImagePath);
    spdlog::error("STB Error: {}", stbi_failure_reason());
  } else {
    spdlog::info("Successfully loaded image: {}", kImagePath);
    spdlog::debug("  Dimensions: {}x{}", width_, height_);
    spdlog::debug("  Channels: {}", channels_);
    spdlog::debug("  Total bytes: {}", width_ * height_ * channels_);
  }
}

ImageLoader::ImageLoader(const char* path) 
    : data_(nullptr), width_(0), height_(0), channels_(0) {
  data_ = stbi_load(path, &width_, &height_, &channels_, 0);
  
  if (data_ == nullptr) {
    spdlog::error("Failed to load image from {}", path);
    spdlog::error("STB Error: {}", stbi_failure_reason());
  } else {
    spdlog::info("Successfully loaded image: {}", path);
    spdlog::debug("  Dimensions: {}x{}", width_, height_);
    spdlog::debug("  Channels: {}", channels_);
    spdlog::debug("  Total bytes: {}", width_ * height_ * channels_);
  }
}

ImageLoader::~ImageLoader() {
  if (data_ != nullptr) {
    stbi_image_free(data_);
    data_ = nullptr;
  }
}

ImageLoader::ImageLoader(ImageLoader&& other) noexcept
    : data_(other.data_),
      width_(other.width_),
      height_(other.height_),
      channels_(other.channels_) {
  other.data_ = nullptr;
  other.width_ = 0;
  other.height_ = 0;
  other.channels_ = 0;
}

ImageLoader& ImageLoader::operator=(ImageLoader&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) {
      stbi_image_free(data_);
    }
    
    data_ = other.data_;
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    
    other.data_ = nullptr;
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
  }
  return *this;
}

}  // namespace jrb::infrastructure::image

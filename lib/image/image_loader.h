#pragma once

#include <string>
#include <memory>
#include "interfaces/image_source.h"

namespace jrb::lib::image {

class ImageLoader : public interfaces::IImageSource {
 public:
  ImageLoader();
  explicit ImageLoader(const char* path);
  ~ImageLoader() override;
  
  ImageLoader(const ImageLoader&) = delete;
  ImageLoader& operator=(const ImageLoader&) = delete;
  
  ImageLoader(ImageLoader&& other) noexcept;
  ImageLoader& operator=(ImageLoader&& other) noexcept;
  
  int width() const override { return width_; }
  int height() const override { return height_; }
  int channels() const override { return channels_; }
  const unsigned char* data() const override { return data_; }
  bool is_valid() const override { return data_ != nullptr; }
  
  bool is_loaded() const { return is_valid(); }
  
  static constexpr const char* kImagePath = "data/lena.png";
  
 private:
  unsigned char* data_;
  int width_;
  int height_;
  int channels_;
};

}  // namespace jrb::lib::image

#pragma once

#include "cpp_accelerator/domain/interfaces/image_sink.h"

namespace jrb::infrastructure::image {

class ImageWriter : public domain::interfaces::IImageSink {
public:
  ImageWriter() = default;
  ~ImageWriter() override = default;

  ImageWriter(const ImageWriter&) = delete;
  ImageWriter& operator=(const ImageWriter&) = delete;

  bool write(const char* filepath, const unsigned char* data, int width, int height,
             int channels) override;
};

}  // namespace jrb::infrastructure::image

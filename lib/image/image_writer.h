#pragma once

#include "interfaces/image_sink.h"

namespace jrb::lib::image {

class ImageWriter : public interfaces::IImageSink {
 public:
  ImageWriter() = default;
  ~ImageWriter() override = default;
  
  ImageWriter(const ImageWriter&) = delete;
  ImageWriter& operator=(const ImageWriter&) = delete;
  
  bool write(const char* filepath,
            const unsigned char* data,
            int width,
            int height,
            int channels) override;
};

}  // namespace jrb::lib::image

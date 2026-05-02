#pragma once

#include <cstdint>
#include <vector>

#include "src/cpp_accelerator/domain/interfaces/image_sink.h"

namespace jrb::adapters::image {

class ImageWriter : public domain::interfaces::IImageSink {
public:
  ImageWriter() = default;
  ~ImageWriter() override = default;

  ImageWriter(const ImageWriter&) = delete;
  ImageWriter& operator=(const ImageWriter&) = delete;

  bool write(const char* filepath, const unsigned char* data, int width, int height,
             int channels) override;

  bool writeBmp(const char* filepath, const unsigned char* data, int width, int height,
                int channels) override;

  bool readAsPng(const char* filepath, std::vector<uint8_t>* png_data,
                 int* width, int* height,
                 int max_width, int max_height) override;
};

}  // namespace jrb::adapters::image

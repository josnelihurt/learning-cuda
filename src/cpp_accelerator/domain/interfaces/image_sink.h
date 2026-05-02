#pragma once

#include <cstdint>
#include <vector>

namespace jrb::domain::interfaces {

// Interface for writing image data
class IImageSink {
public:
  virtual ~IImageSink() = default;

  virtual bool write(const char* filepath, const unsigned char* data, int width, int height,
                     int channels) = 0;

  virtual bool writeBmp(const char* filepath, const unsigned char* data, int width, int height,
                        int channels) = 0;

  // Load an image from filepath (BMP/PNG/JPEG), encode as PNG, and write bytes into png_data.
  // Returns actual output dimensions in *width/*height.
  // If max_width/max_height > 0 and the image is larger, nearest-neighbor downsampling is applied.
  // Pass 0 for max_width/max_height to return full resolution.
  virtual bool readAsPng(const char* filepath, std::vector<uint8_t>* png_data,
                         int* width, int* height,
                         int max_width, int max_height) = 0;
};

}  // namespace jrb::domain::interfaces

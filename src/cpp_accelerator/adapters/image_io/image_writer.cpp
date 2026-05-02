#include "src/cpp_accelerator/adapters/image_io/image_writer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::adapters::image {

bool ImageWriter::write(const char* filepath, const unsigned char* data, int width, int height,
                        int channels) {
  if (data == nullptr || width <= 0 || height <= 0 || channels <= 0) {
    spdlog::error("Invalid image data for writing");
    return false;
  }

  int result = stbi_write_png(filepath, width, height, channels, data, width * channels);

  if (result == 0) {
    spdlog::error("Failed to write image to {}", filepath);
    return false;
  }

  spdlog::info("Successfully wrote image: {}", filepath);
  spdlog::debug("  Dimensions: {}x{}", width, height);
  spdlog::debug("  Channels: {}", channels);

  return true;
}

bool ImageWriter::writeBmp(const char* filepath, const unsigned char* data, int width, int height,
                           int channels) {
  if (data == nullptr || width <= 0 || height <= 0 || channels <= 0) {
    spdlog::error("Invalid image data for BMP writing");
    return false;
  }

  int result = stbi_write_bmp(filepath, width, height, channels, data);

  if (result == 0) {
    spdlog::error("Failed to write BMP to {}", filepath);
    return false;
  }

  spdlog::info("Wrote BMP capture: {}", filepath);
  return true;
}

}  // namespace jrb::adapters::image

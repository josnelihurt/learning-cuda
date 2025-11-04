#include "cpp_accelerator/infrastructure/image/image_writer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmissing-requires"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::infrastructure::image {

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

}  // namespace jrb::infrastructure::image

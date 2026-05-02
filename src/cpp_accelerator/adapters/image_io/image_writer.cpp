#include "src/cpp_accelerator/adapters/image_io/image_writer.h"

#include <algorithm>
#include <cstring>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#pragma GCC diagnostic pop

// stb_image.h implementation lives in image_loader.cpp (same cc_library).
// Include the header here for declarations only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "third_party/stb/stb_image.h"
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::adapters::image {

using std::vector;

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

bool ImageWriter::readAsPng(const char* filepath, vector<uint8_t>* png_data,
                             int* width, int* height,
                             int max_width, int max_height) {
  int channels = 0;
  unsigned char* data = stbi_load(filepath, width, height, &channels, 3);
  if (data == nullptr) {
    spdlog::error("readAsPng: failed to load {}: {}", filepath, stbi_failure_reason());
    return false;
  }

  int out_w = *width;
  int out_h = *height;
  vector<uint8_t> scaled;
  const unsigned char* src = data;

  if ((max_width > 0 && *width > max_width) || (max_height > 0 && *height > max_height)) {
    float scale = 1.0F;
    if (max_width > 0) {
      scale = std::max(scale, static_cast<float>(*width) / static_cast<float>(max_width));
    }
    if (max_height > 0) {
      scale = std::max(scale, static_cast<float>(*height) / static_cast<float>(max_height));
    }
    out_w = static_cast<int>(static_cast<float>(*width)  / scale);
    out_h = static_cast<int>(static_cast<float>(*height) / scale);
    scaled.resize(static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3);
    for (int y = 0; y < out_h; ++y) {
      for (int x = 0; x < out_w; ++x) {
        const int sx = static_cast<int>(static_cast<float>(x) * scale);
        const int sy = static_cast<int>(static_cast<float>(y) * scale);
        std::memcpy(&scaled[static_cast<size_t>(y * out_w + x) * 3],
                    &data[static_cast<size_t>(sy * (*width) + sx) * 3], 3);
      }
    }
    src = scaled.data();
    *width  = out_w;
    *height = out_h;
  }

  stbi_write_png_to_func(
      [](void* ctx, void* buf, int len) {
        auto* out = static_cast<vector<uint8_t>*>(ctx);
        const auto* b = static_cast<uint8_t*>(buf);
        out->insert(out->end(), b, b + len);
      },
      png_data, out_w, out_h, 3, src, out_w * 3);

  stbi_image_free(data);

  if (png_data->empty()) {
    spdlog::error("readAsPng: PNG encoding produced no output for {}", filepath);
    return false;
  }
  spdlog::debug("readAsPng: {} → {}×{} {} bytes PNG", filepath, out_w, out_h, png_data->size());
  return true;
}

}  // namespace jrb::adapters::image

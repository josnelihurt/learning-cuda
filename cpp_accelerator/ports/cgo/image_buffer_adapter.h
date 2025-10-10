#pragma once

#include <cstdint>
#include <vector>

#include "cpp_accelerator/domain/interfaces/image_sink.h"
#include "cpp_accelerator/domain/interfaces/image_source.h"

namespace jrb::ports::cgo {

// Adapts raw image buffer to IImageSource interface
class ImageBufferSource final : public domain::interfaces::IImageSource {
   public:
    ImageBufferSource(const uint8_t* data, int width, int height, int channels);
    ~ImageBufferSource() override = default;

    int width() const override {
        return width_;
    }
    int height() const override {
        return height_;
    }
    int channels() const override {
        return channels_;
    }
    const unsigned char* data() const override {
        return data_;
    }
    bool is_valid() const override {
        return data_ != nullptr && width_ > 0 && height_ > 0;
    }

   private:
    const uint8_t* data_;
    int width_;
    int height_;
    int channels_;
};

// Adapts IImageSink to capture processed image data
class ImageBufferSink final : public domain::interfaces::IImageSink {
   public:
    ImageBufferSink();
    ~ImageBufferSink() override = default;

    bool write(const char* filepath, const unsigned char* data, int width, int height,
               int channels) override;

    // Accessors for processed data
    const std::vector<uint8_t>& get_data() const {
        return buffer_;
    }
    int width() const {
        return width_;
    }
    int height() const {
        return height_;
    }
    int channels() const {
        return channels_;
    }

   private:
    std::vector<uint8_t> buffer_;
    int width_;
    int height_;
    int channels_;
};

}  // namespace jrb::ports::cgo

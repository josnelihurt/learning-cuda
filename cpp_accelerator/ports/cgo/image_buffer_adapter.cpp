#include "cpp_accelerator/ports/cgo/image_buffer_adapter.h"

#include <cstring>

namespace jrb::ports::cgo {

ImageBufferSource::ImageBufferSource(const uint8_t* data, int width, int height, int channels)
    : data_(data), width_(width), height_(height), channels_(channels) {
}

ImageBufferSink::ImageBufferSink() : width_(0), height_(0), channels_(0) {
}

bool ImageBufferSink::write(const char* filepath, const unsigned char* data, int width, int height,
                            int channels) {
    // Ignore filepath parameter - we're capturing in memory, not writing to disk
    if (data == nullptr || width <= 0 || height <= 0 || channels <= 0) {
        return false;
    }

    width_ = width;
    height_ = height;
    channels_ = channels;

    size_t data_size = static_cast<size_t>(width) * height * channels;
    buffer_.resize(data_size);
    std::memcpy(buffer_.data(), data, data_size);

    return true;
}

}  // namespace jrb::ports::cgo

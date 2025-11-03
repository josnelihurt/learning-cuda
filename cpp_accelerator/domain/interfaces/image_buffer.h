#pragma once

namespace jrb::domain::interfaces {

struct ImageBuffer {
  const unsigned char* data;
  int width;
  int height;
  int channels;

  ImageBuffer(const unsigned char* buffer_data, int w, int h, int ch)
      : data(buffer_data), width(w), height(h), channels(ch) {}

  bool IsValid() const { return data != nullptr && width > 0 && height > 0 && channels > 0; }

  int GetTotalSize() const { return width * height * channels; }
};

struct ImageBufferMut {
  unsigned char* data;
  int width;
  int height;
  int channels;

  ImageBufferMut(unsigned char* buffer_data, int w, int h, int ch)
      : data(buffer_data), width(w), height(h), channels(ch) {}

  bool IsValid() const { return data != nullptr && width > 0 && height > 0 && channels > 0; }

  int GetTotalSize() const { return width * height * channels; }
};

struct FilterContext {
  ImageBuffer input;
  ImageBufferMut output;

  FilterContext(const unsigned char* input_data, unsigned char* output_data, int width, int height,
                int channels)
      : input(input_data, width, height, channels), output(output_data, width, height, channels) {}
};

}  // namespace jrb::domain::interfaces

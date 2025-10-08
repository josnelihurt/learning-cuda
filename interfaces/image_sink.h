#pragma once

namespace jrb::interfaces {

// Interface for writing image data
class IImageSink {
 public:
  virtual ~IImageSink() = default;
  
  virtual bool write(const char* filepath, 
                    const unsigned char* data,
                    int width, 
                    int height, 
                    int channels) = 0;
};

}  // namespace jrb::interfaces

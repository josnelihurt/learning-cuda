#pragma once

namespace jrb::domain::interfaces {

class IImageSource {
 public:
  virtual ~IImageSource() = default;
  
  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual int channels() const = 0;
  virtual const unsigned char* data() const = 0;
  virtual bool is_valid() const = 0;
};

}  // namespace jrb::domain::interfaces

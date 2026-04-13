#pragma once

#include "cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::domain::interfaces {

class IPixelGetter {
public:
  virtual ~IPixelGetter() = default;

  virtual float GetPixelValue(const ImageBuffer& buffer, int x, int y) const = 0;
};

}  // namespace jrb::domain::interfaces

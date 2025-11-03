#pragma once

#include "cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::domain::interfaces {

enum class FilterType { GRAYSCALE, BLUR };

class IFilter {
public:
  virtual ~IFilter() = default;

  virtual bool Apply(FilterContext& context) = 0;

  virtual FilterType GetType() const = 0;

  virtual bool IsInPlace() const = 0;
};

}  // namespace jrb::domain::interfaces

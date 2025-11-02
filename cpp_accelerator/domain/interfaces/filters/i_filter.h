#pragma once

namespace jrb::domain::interfaces {

enum class FilterType { GRAYSCALE, BLUR };

struct FilterContext {
  const unsigned char* input_data;
  unsigned char* output_data;
  int width;
  int height;
  int channels;
};

class IFilter {
public:
  virtual ~IFilter() = default;

  virtual bool Apply(const FilterContext& context) = 0;

  virtual FilterType GetType() const = 0;

  virtual bool IsInPlace() const = 0;
};

}  // namespace jrb::domain::interfaces

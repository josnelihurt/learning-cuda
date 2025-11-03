#pragma once

#include "cpp_accelerator/domain/interfaces/filters/i_filter.h"

namespace jrb::infrastructure::cpu {

enum class GrayscaleAlgorithm { BT601, BT709, Average, Lightness, Luminosity };

class GrayscaleFilter : public jrb::domain::interfaces::IFilter {
public:
  explicit GrayscaleFilter(GrayscaleAlgorithm algorithm = GrayscaleAlgorithm::BT601);
  ~GrayscaleFilter() override = default;

  bool Apply(jrb::domain::interfaces::FilterContext& context) override;

  jrb::domain::interfaces::FilterType GetType() const override;

  bool IsInPlace() const override;

  void SetAlgorithm(GrayscaleAlgorithm algorithm);
  GrayscaleAlgorithm GetAlgorithm() const;

private:
  unsigned char CalculateGrayscaleValue(unsigned char r, unsigned char g, unsigned char b) const;

  GrayscaleAlgorithm algorithm_;
};

}  // namespace jrb::infrastructure::cpu

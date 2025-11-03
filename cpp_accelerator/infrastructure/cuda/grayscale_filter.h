#pragma once

#include "cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "cpp_accelerator/infrastructure/cuda/grayscale_processor.h"

namespace jrb::infrastructure::cuda {

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
  GrayscaleAlgorithm algorithm_;
};

}  // namespace jrb::infrastructure::cuda

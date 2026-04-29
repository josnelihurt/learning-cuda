#pragma once

#include <memory>

#include "src/cpp_accelerator/domain/interfaces/filters/i_filter.h"

namespace jrb::application::engine {

template <typename OnGrayscale, typename OnBlur>
std::unique_ptr<jrb::domain::interfaces::IFilter> DispatchCreateFilter(
    jrb::domain::interfaces::FilterType type, OnGrayscale&& on_grayscale, OnBlur&& on_blur) {
  using jrb::domain::interfaces::FilterType;

  switch (type) {
    case FilterType::GRAYSCALE:
      return on_grayscale();
    case FilterType::BLUR:
      return on_blur();
    default:
      return nullptr;
  }
}

}  // namespace jrb::application::engine

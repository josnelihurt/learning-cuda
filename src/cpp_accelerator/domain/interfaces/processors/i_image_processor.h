#pragma once

#include <string>
#include "src/cpp_accelerator/domain/interfaces/image_sink.h"
#include "src/cpp_accelerator/domain/interfaces/image_source.h"

namespace jrb::domain::interfaces {

class IImageProcessor {
public:
  virtual ~IImageProcessor() = default;

  virtual bool process(IImageSource& source, IImageSink& sink, const std::string& output_path) = 0;
};

}  // namespace jrb::domain::interfaces

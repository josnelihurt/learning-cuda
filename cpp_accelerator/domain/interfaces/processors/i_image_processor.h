#pragma once

#include "cpp_accelerator/domain/interfaces/image_source.h"
#include "cpp_accelerator/domain/interfaces/image_sink.h"
#include <string>

namespace jrb::domain::interfaces {

class IImageProcessor {
public:
    virtual ~IImageProcessor() = default;
    
    virtual bool process(
        IImageSource& source,
        IImageSink& sink,
        const std::string& output_path
    ) = 0;
};

}  // namespace jrb::domain::interfaces


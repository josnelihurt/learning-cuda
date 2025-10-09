#pragma once

#include "domain/interfaces/image_source.h"
#include "domain/interfaces/image_sink.h"
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


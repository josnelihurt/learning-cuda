#pragma once

#include "domain/interfaces/image_source.h"
#include "domain/interfaces/image_sink.h"
#include "domain/interfaces/processors/i_image_processor.h"
#include <string>

namespace jrb::infrastructure::cuda {

class GrayscaleProcessor : public domain::interfaces::IImageProcessor {
 public:
  GrayscaleProcessor() = default;
  ~GrayscaleProcessor() override = default;
  
  GrayscaleProcessor(const GrayscaleProcessor&) = delete;
  GrayscaleProcessor& operator=(const GrayscaleProcessor&) = delete;
  
  bool process(domain::interfaces::IImageSource& source, 
               domain::interfaces::IImageSink& sink, 
               const std::string& output_path) override;
  
 private:
  void convert_to_grayscale_cuda(const unsigned char* input,
                                 unsigned char* output,
                                 int width,
                                 int height,
                                 int channels);
};

}  // namespace jrb::infrastructure::cuda

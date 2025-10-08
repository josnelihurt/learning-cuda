#pragma once

#include "interfaces/image_source.h"
#include "interfaces/image_sink.h"

namespace jrb::lib::cuda {

class GrayscaleProcessor {
 public:
  GrayscaleProcessor() = default;
  ~GrayscaleProcessor() = default;
  
  GrayscaleProcessor(const GrayscaleProcessor&) = delete;
  GrayscaleProcessor& operator=(const GrayscaleProcessor&) = delete;
  
  bool process(interfaces::IImageSource& source, 
               interfaces::IImageSink& sink, 
               const char* output_path);
  
 private:
  void convert_to_grayscale_cuda(const unsigned char* input,
                                 unsigned char* output,
                                 int width,
                                 int height,
                                 int channels);
};

}  // namespace jrb::lib::cuda

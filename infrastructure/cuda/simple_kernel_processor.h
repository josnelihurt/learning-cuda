#pragma once

#include "domain/interfaces/processors/i_image_processor.h"

namespace jrb::infrastructure::cuda {

class SimpleKernelProcessor final: public domain::interfaces::IImageProcessor {
 public:
  SimpleKernelProcessor() = default;
  ~SimpleKernelProcessor() override = default;
  
  SimpleKernelProcessor(const SimpleKernelProcessor&) = delete;
  SimpleKernelProcessor& operator=(const SimpleKernelProcessor&) = delete;
  
  bool process(domain::interfaces::IImageSource& source, 
               domain::interfaces::IImageSink& sink, 
               const std::string& output_path) override;
};

}  // namespace jrb::infrastructure::cuda


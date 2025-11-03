#pragma once

// TODO(migration): DECIDE: Keep this file for internal commands only OR delete completely.
// This processor is used in:
// - cpp_accelerator/application/commands/command_factory.cpp (internal CLI commands)
// - cpp_accelerator/ports/shared_lib/cuda_processor_impl.cpp (TO BE REMOVED - use FilterPipeline)
// - cpp_accelerator/ports/cgo/cgo_api.cpp (TO BE REMOVED - use FilterPipeline)
//
// Options:
// A) Keep for internal commands only (command_factory.cpp). Remove from public API.
// B) Delete completely and refactor command_factory.cpp to use FilterPipeline too.
//
// Recommendation: Option B - migrate everything to FilterPipeline for consistency.
// See migration TODOs in cpp_accelerator/ports/shared_lib/cuda_processor_impl.cpp (line 24)

#include <string>

#include "cpp_accelerator/domain/interfaces/image_sink.h"
#include "cpp_accelerator/domain/interfaces/image_source.h"
#include "cpp_accelerator/domain/interfaces/processors/i_image_processor.h"

namespace jrb::infrastructure::cpu {

// TODO(migration): Move this enum to domain/interfaces to be shared by CPU and CUDA filters
enum class GrayscaleAlgorithm {
  BT601,      // ITU-R BT.601 (SDTV): Y = 0.299R + 0.587G + 0.114B
  BT709,      // ITU-R BT.709 (HDTV): Y = 0.2126R + 0.7152G + 0.0722B
  Average,    // Simple average: Y = (R + G + B) / 3
  Lightness,  // Lightness: Y = (max(R,G,B) + min(R,G,B)) / 2
  Luminosity  // Luminosity: Y = 0.21R + 0.72G + 0.07B
};

class CpuGrayscaleProcessor : public domain::interfaces::IImageProcessor {
public:
  explicit CpuGrayscaleProcessor(GrayscaleAlgorithm algorithm = GrayscaleAlgorithm::BT601);
  ~CpuGrayscaleProcessor() override = default;

  bool process(domain::interfaces::IImageSource& source, domain::interfaces::IImageSink& sink,
               const std::string& output_path) override;

  void set_algorithm(GrayscaleAlgorithm algorithm);
  GrayscaleAlgorithm get_algorithm() const { return algorithm_; }

private:
  void convert_to_grayscale_cpu(const unsigned char* input, unsigned char* output, int width,
                                int height, int channels);

  unsigned char calculate_grayscale_value(unsigned char r, unsigned char g, unsigned char b) const;

  GrayscaleAlgorithm algorithm_;
};

}  // namespace jrb::infrastructure::cpu

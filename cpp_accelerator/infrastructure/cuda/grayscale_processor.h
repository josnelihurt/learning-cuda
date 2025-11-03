#pragma once

// TODO(migration): DELETE THIS FILE after implementing CudaGrayscaleFilter.
// Migration plan:
// 1. Create cpp_accelerator/infrastructure/cuda/grayscale_filter.h implementing IFilter
// 2. Move GrayscaleAlgorithm enum to domain/interfaces (unified for CPU and CUDA)
// 3. Implement CudaGrayscaleFilter::Apply() using existing CUDA kernels from grayscale_processor.cu
// 4. Update all code using GrayscaleProcessor to use CudaGrayscaleFilter via FilterPipeline
// 5. Keep this file only if GrayscaleProcessor is still needed for internal commands
//    (see command_factory.cpp). Otherwise delete completely.
// See migration TODOs in:
// - cpp_accelerator/ports/shared_lib/cuda_processor_impl.cpp (line 24)
// - cpp_accelerator/ports/cgo/cgo_api.cpp

#include <string>

#include "cpp_accelerator/domain/interfaces/image_sink.h"
#include "cpp_accelerator/domain/interfaces/image_source.h"
#include "cpp_accelerator/domain/interfaces/processors/i_image_processor.h"

namespace jrb::infrastructure::cuda {

// TODO(migration): Move this enum to domain/interfaces to be shared by CPU and CUDA filters
enum class GrayscaleAlgorithm {
  BT601,      // ITU-R BT.601 (SDTV): Y = 0.299R + 0.587G + 0.114B
  BT709,      // ITU-R BT.709 (HDTV): Y = 0.2126R + 0.7152G + 0.0722B
  Average,    // Simple average: Y = (R + G + B) / 3
  Lightness,  // Lightness: Y = (max(R,G,B) + min(R,G,B)) / 2
  Luminosity  // Luminosity: Y = 0.21R + 0.72G + 0.07B
};

class GrayscaleProcessor final : public domain::interfaces::IImageProcessor {
public:
  explicit GrayscaleProcessor(GrayscaleAlgorithm algorithm = GrayscaleAlgorithm::BT601);
  ~GrayscaleProcessor() override = default;

  GrayscaleProcessor(const GrayscaleProcessor&) = delete;
  GrayscaleProcessor& operator=(const GrayscaleProcessor&) = delete;

  bool process(domain::interfaces::IImageSource& source, domain::interfaces::IImageSink& sink,
               const std::string& output_path) override;

  void set_algorithm(GrayscaleAlgorithm algorithm);
  GrayscaleAlgorithm get_algorithm() const { return algorithm_; }

private:
  void convert_to_grayscale_cuda(const unsigned char* input, unsigned char* output, int width,
                                 int height, int channels);

  GrayscaleAlgorithm algorithm_;
};

}  // namespace jrb::infrastructure::cuda

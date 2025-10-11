#include "cpp_accelerator/infrastructure/cpu/grayscale_processor.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>

namespace jrb::infrastructure::cpu {

CpuGrayscaleProcessor::CpuGrayscaleProcessor(GrayscaleAlgorithm algorithm) : algorithm_(algorithm) {
}

void CpuGrayscaleProcessor::set_algorithm(GrayscaleAlgorithm algorithm) {
    algorithm_ = algorithm;
}

unsigned char CpuGrayscaleProcessor::calculate_grayscale_value(unsigned char r, unsigned char g,
                                                               unsigned char b) const {
    switch (algorithm_) {
        case GrayscaleAlgorithm::BT601:
            // ITU-R BT.601 (SDTV): Y = 0.299R + 0.587G + 0.114B
            return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        case GrayscaleAlgorithm::BT709:
            // ITU-R BT.709 (HDTV): Y = 0.2126R + 0.7152G + 0.0722B
            return static_cast<unsigned char>(0.2126f * r + 0.7152f * g + 0.0722f * b);

        case GrayscaleAlgorithm::Average:
            // Simple average: Y = (R + G + B) / 3
            return static_cast<unsigned char>(
                (static_cast<int>(r) + static_cast<int>(g) + static_cast<int>(b)) / 3);

        case GrayscaleAlgorithm::Lightness:
            // Lightness: Y = (max(R,G,B) + min(R,G,B)) / 2
            {
                unsigned char max_val = std::max({r, g, b});
                unsigned char min_val = std::min({r, g, b});
                return static_cast<unsigned char>(
                    (static_cast<int>(max_val) + static_cast<int>(min_val)) / 2);
            }

        case GrayscaleAlgorithm::Luminosity:
            // Luminosity: Y = 0.21R + 0.72G + 0.07B
            return static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

        default:
            // Default to BT601
            return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void CpuGrayscaleProcessor::convert_to_grayscale_cpu(const unsigned char* input,
                                                     unsigned char* output, int width, int height,
                                                     int channels) {
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        int input_idx = i * channels;

        if (channels >= 3) {
            unsigned char r = input[input_idx];
            unsigned char g = input[input_idx + 1];
            unsigned char b = input[input_idx + 2];

            output[i] = calculate_grayscale_value(r, g, b);
        } else {
            // Already grayscale or single channel
            output[i] = input[input_idx];
        }
    }
}

bool CpuGrayscaleProcessor::process(domain::interfaces::IImageSource& source,
                                    domain::interfaces::IImageSink& sink,
                                    const std::string& output_path) {
    if (!source.is_valid()) {
        spdlog::error("Invalid image source");
        return false;
    }

    int width = source.width();
    int height = source.height();
    int channels = source.channels();
    const unsigned char* input_data = source.data();

    std::unique_ptr<unsigned char[]> output_data(new unsigned char[width * height]);

    convert_to_grayscale_cpu(input_data, output_data.get(), width, height, channels);

    return sink.write(output_path.c_str(), output_data.get(), width, height, 1);
}

}  // namespace jrb::infrastructure::cpu

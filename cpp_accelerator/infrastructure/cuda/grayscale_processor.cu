#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <memory>

#include "cpp_accelerator/infrastructure/cuda/grayscale_processor.h"

namespace jrb::infrastructure::cuda {

// Grayscale algorithm type (passed to kernel)
enum class GrayscaleAlgorithmType : int {
    BT601 = 0,
    BT709 = 1,
    Average = 2,
    Lightness = 3,
    Luminosity = 4
};

// Calculate grayscale value based on algorithm
__device__ unsigned char calculate_grayscale_value(unsigned char r, unsigned char g,
                                                   unsigned char b,
                                                   GrayscaleAlgorithmType algorithm) {
    switch (algorithm) {
        case GrayscaleAlgorithmType::BT601:
            // ITU-R BT.601 (SDTV): Y = 0.299R + 0.587G + 0.114B
            return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        case GrayscaleAlgorithmType::BT709:
            // ITU-R BT.709 (HDTV): Y = 0.2126R + 0.7152G + 0.0722B
            return static_cast<unsigned char>(0.2126f * r + 0.7152f * g + 0.0722f * b);

        case GrayscaleAlgorithmType::Average:
            // Simple average: Y = (R + G + B) / 3
            return static_cast<unsigned char>((r + g + b) / 3);

        case GrayscaleAlgorithmType::Lightness:
            // Lightness: Y = (max(R,G,B) + min(R,G,B)) / 2
            {
                unsigned char max_val = max(r, max(g, b));
                unsigned char min_val = min(r, min(g, b));
                return static_cast<unsigned char>((max_val + min_val) / 2);
            }

        case GrayscaleAlgorithmType::Luminosity:
            // Luminosity: Y = 0.21R + 0.72G + 0.07B
            return static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

        default:
            // Default to BT601
            return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Legacy function for backwards compatibility
__device__ unsigned char calculate_luminosity(unsigned char r, unsigned char g, unsigned char b) {
    return calculate_grayscale_value(r, g, b, GrayscaleAlgorithmType::BT601);
}

__device__ bool is_within_bounds(int x, int y, int width, int height) {
    return x < width && y < height;
}

__device__ int calculate_pixel_index(int x, int y, int width) {
    return y * width + x;
}

__global__ void convert_to_grayscale_kernel(const unsigned char* input, unsigned char* output,
                                            int width, int height, int channels,
                                            GrayscaleAlgorithmType algorithm) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!is_within_bounds(x, y, width, height)) {
        return;
    }

    int pixel_idx = calculate_pixel_index(x, y, width);
    int input_idx = pixel_idx * channels;

    if (channels >= 3) {
        unsigned char r = input[input_idx];
        unsigned char g = input[input_idx + 1];
        unsigned char b = input[input_idx + 2];

        output[pixel_idx] = calculate_grayscale_value(r, g, b, algorithm);
    } else {
        output[pixel_idx] = input[input_idx];
    }
}

size_t calculate_image_buffer_size(int width, int height, int channels) {
    return width * height * channels;
}

size_t calculate_grayscale_buffer_size(int width, int height) {
    return width * height;
}

dim3 calculate_optimal_block_size() {
    return dim3(16, 16);
}

dim3 calculate_grid_size(int width, int height, dim3 block_size) {
    return dim3((width + block_size.x - 1) / block_size.x,
                (height + block_size.y - 1) / block_size.y);
}

void print_kernel_launch_info(dim3 grid_size, dim3 block_size) {
    spdlog::debug("Launching CUDA kernel:");
    spdlog::debug("  Grid size: {}x{}", grid_size.x, grid_size.y);
    spdlog::debug("  Block size: {}x{}", block_size.x, block_size.y);
    spdlog::debug("  Total threads: {}", grid_size.x * block_size.x * grid_size.y * block_size.y);
}

GrayscaleProcessor::GrayscaleProcessor(GrayscaleAlgorithm algorithm) : algorithm_(algorithm) {
}

void GrayscaleProcessor::set_algorithm(GrayscaleAlgorithm algorithm) {
    algorithm_ = algorithm;
}

void GrayscaleProcessor::convert_to_grayscale_cuda(const unsigned char* input,
                                                   unsigned char* output, int width, int height,
                                                   int channels) {
    size_t input_size = calculate_image_buffer_size(width, height, channels);
    size_t output_size = calculate_grayscale_buffer_size(width, height);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    dim3 block_size = calculate_optimal_block_size();
    dim3 grid_size = calculate_grid_size(width, height, block_size);

    print_kernel_launch_info(grid_size, block_size);

    // Convert algorithm enum to kernel type
    GrayscaleAlgorithmType kernel_algorithm = static_cast<GrayscaleAlgorithmType>(algorithm_);

    convert_to_grayscale_kernel<<<grid_size, block_size>>>(d_input, d_output, width, height,
                                                           channels, kernel_algorithm);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        spdlog::error("CUDA kernel error: {}", cudaGetErrorString(error));
    }
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

bool GrayscaleProcessor::process(domain::interfaces::IImageSource& source,
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

    spdlog::info("Processing image to grayscale (CUDA)...");
    spdlog::info("  Input: {}x{} ({} channels)", width, height, channels);
    spdlog::info("  Algorithm: {}", static_cast<int>(algorithm_));

    std::unique_ptr<unsigned char[]> output_data(new unsigned char[width * height]);

    convert_to_grayscale_cuda(input_data, output_data.get(), width, height, channels);

    spdlog::info("  Output: {}x{} (1 channel)", width, height);

    return sink.write(output_path.c_str(), output_data.get(), width, height, 1);
}

}  // namespace jrb::infrastructure::cuda

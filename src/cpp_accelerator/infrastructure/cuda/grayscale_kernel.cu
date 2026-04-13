#include <cuda_runtime.h>

namespace jrb::infrastructure::cuda {

enum class GrayscaleAlgorithmType : int {
  BT601 = 0,
  BT709 = 1,
  Average = 2,
  Lightness = 3,
  Luminosity = 4
};

__device__ unsigned char calculate_grayscale_value(unsigned char r, unsigned char g,
                                                   unsigned char b,
                                                   GrayscaleAlgorithmType algorithm) {
  switch (algorithm) {
    case GrayscaleAlgorithmType::BT601:
      return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    case GrayscaleAlgorithmType::BT709:
      return static_cast<unsigned char>(0.2126f * r + 0.7152f * g + 0.0722f * b);
    case GrayscaleAlgorithmType::Average:
      return static_cast<unsigned char>((r + g + b) / 3);
    case GrayscaleAlgorithmType::Lightness: {
      unsigned char max_val = max(r, max(g, b));
      unsigned char min_val = min(r, min(g, b));
      return static_cast<unsigned char>((max_val + min_val) / 2);
    }
    case GrayscaleAlgorithmType::Luminosity:
      return static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);
    default:
      return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
  }
}

__global__ void convert_to_grayscale_kernel(const unsigned char* input, unsigned char* output,
                                            int width, int height, int channels,
                                            GrayscaleAlgorithmType algorithm) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int pixel_idx = y * width + x;
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

// Pure CUDA function without spdlog dependencies
extern "C" cudaError_t cuda_convert_to_grayscale(const unsigned char* input, unsigned char* output,
                                                 int width, int height, int channels,
                                                 int algorithm) {
  size_t input_size = width * height * channels;
  size_t output_size = width * height;

  unsigned char* d_input = nullptr;
  unsigned char* d_output = nullptr;

  // Allocate device memory
  cudaError_t error = cudaMalloc(&d_input, input_size);
  if (error != cudaSuccess) {
    return error;
  }

  error = cudaMalloc(&d_output, output_size);
  if (error != cudaSuccess) {
    cudaFree(d_input);
    return error;
  }

  // Copy input to device
  error = cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_output);
    return error;
  }

  // Launch kernel
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);

  GrayscaleAlgorithmType kernel_algorithm = static_cast<GrayscaleAlgorithmType>(algorithm);
  convert_to_grayscale_kernel<<<grid_size, block_size>>>(d_input, d_output, width, height, channels,
                                                         kernel_algorithm);

  // Synchronize and check for errors
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_output);
    return error;
  }

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_output);
    return error;
  }

  // Copy output to host
  error = cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_output);
    return error;
  }

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);

  return cudaSuccess;
}

}  // namespace jrb::infrastructure::cuda

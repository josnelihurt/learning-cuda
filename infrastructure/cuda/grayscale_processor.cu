#include "infrastructure/cuda/grayscale_processor.h"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <memory>

namespace jrb::infrastructure::cuda {

__device__ unsigned char calculate_luminosity(unsigned char r, unsigned char g, unsigned char b) {
  return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

__device__ bool is_within_bounds(int x, int y, int width, int height) {
  return x < width && y < height;
}

__device__ int calculate_pixel_index(int x, int y, int width) {
  return y * width + x;
}

__global__ void convert_to_grayscale_kernel(const unsigned char* input,
                                           unsigned char* output,
                                           int width,
                                           int height,
                                           int channels) {
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
    
    output[pixel_idx] = calculate_luminosity(r, g, b);
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
  return dim3(
      (width + block_size.x - 1) / block_size.x,
      (height + block_size.y - 1) / block_size.y
  );
}

void print_kernel_launch_info(dim3 grid_size, dim3 block_size) {
  spdlog::debug("Launching CUDA kernel:");
  spdlog::debug("  Grid size: {}x{}", grid_size.x, grid_size.y);
  spdlog::debug("  Block size: {}x{}", block_size.x, block_size.y);
  spdlog::debug("  Total threads: {}", grid_size.x * block_size.x * grid_size.y * block_size.y);
}

void check_cuda_error() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    spdlog::error("CUDA kernel error: {}", cudaGetErrorString(error));
  }
}

void GrayscaleProcessor::convert_to_grayscale_cuda(const unsigned char* input,
                                                   unsigned char* output,
                                                   int width,
                                                   int height,
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
  
  convert_to_grayscale_kernel<<<grid_size, block_size>>>(
      d_input, d_output, width, height, channels
  );
  
  cudaDeviceSynchronize();
  check_cuda_error();
  
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
  
  spdlog::info("Processing image to grayscale...");
  spdlog::info("  Input: {}x{} ({} channels)", width, height, channels);
  
  std::unique_ptr<unsigned char[]> output_data(new unsigned char[width * height]);
  
  convert_to_grayscale_cuda(input_data, output_data.get(), width, height, channels);
  
  spdlog::info("  Output: {}x{} (1 channel)", width, height);
  
  return sink.write(output_path.c_str(), output_data.get(), width, height, 1);
}

}  // namespace jrb::infrastructure::cuda

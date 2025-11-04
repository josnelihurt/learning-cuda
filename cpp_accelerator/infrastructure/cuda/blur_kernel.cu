#include <cuda_runtime.h>

namespace jrb::infrastructure::cuda {

constexpr float SQRT_2PI = 2.506628274631000242F;

enum class BorderMode : int { CLAMP = 0, REFLECT = 1, WRAP = 2 };

__device__ int ClampX(int x, int width, BorderMode mode) {
  switch (mode) {
    case BorderMode::CLAMP:
      return max(0, min(x, width - 1));
    case BorderMode::REFLECT: {
      if (x < 0) {
        return -x - 1;
      }
      if (x >= width) {
        return 2 * width - x - 1;
      }
      return x;
    }
    case BorderMode::WRAP: {
      return ((x % width) + width) % width;
    }
    default:
      return max(0, min(x, width - 1));
  }
}

__device__ int ClampY(int y, int height, BorderMode mode) {
  switch (mode) {
    case BorderMode::CLAMP:
      return max(0, min(y, height - 1));
    case BorderMode::REFLECT: {
      if (y < 0) {
        return -y - 1;
      }
      if (y >= height) {
        return 2 * height - y - 1;
      }
      return y;
    }
    case BorderMode::WRAP: {
      return ((y % height) + height) % height;
    }
    default:
      return max(0, min(y, height - 1));
  }
}

__device__ float GetPixelChannelValue(const unsigned char* data, int x, int y, int width,
                                      int height, int channels, int channel_idx,
                                      BorderMode border_mode) {
  int clamped_x = ClampX(x, width, border_mode);
  int clamped_y = ClampY(y, height, border_mode);

  int index = (clamped_y * width + clamped_x) * channels;
  if (channel_idx < channels) {
    return static_cast<float>(data[index + channel_idx]);
  }
  return 0.0F;
}

__global__ void ApplyHorizontalBlurKernel(const unsigned char* input, unsigned char* output,
                                          int width, int height, int channels, const float* kernel,
                                          int kernel_size, BorderMode border_mode) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= width) {
    return;
  }

  int radius = kernel_size / 2;

  for (int y = 0; y < height; y++) {
    int output_idx = (y * width + x) * channels;

    for (int c = 0; c < channels; c++) {
      float sum = 0.0F;
      for (int k = -radius; k <= radius; k++) {
        int pixel_x = x + k;
        float weight = kernel[k + radius];
        sum += GetPixelChannelValue(input, pixel_x, y, width, height, channels, c, border_mode) *
               weight;
      }
      output[output_idx + c] = static_cast<unsigned char>(max(0.0F, min(sum, 255.0F)));
    }
  }
}

__global__ void ApplyVerticalBlurKernel(const unsigned char* input, unsigned char* output,
                                        int width, int height, int channels, const float* kernel,
                                        int kernel_size, BorderMode border_mode) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= height) {
    return;
  }

  int radius = kernel_size / 2;

  for (int x = 0; x < width; x++) {
    int output_idx = (y * width + x) * channels;

    for (int c = 0; c < channels; c++) {
      float sum = 0.0F;
      for (int k = -radius; k <= radius; k++) {
        int pixel_y = y + k;
        float weight = kernel[k + radius];
        sum += GetPixelChannelValue(input, x, pixel_y, width, height, channels, c, border_mode) *
               weight;
      }
      output[output_idx + c] = static_cast<unsigned char>(max(0.0F, min(sum, 255.0F)));
    }
  }
}

static void cleanup_memory(unsigned char* d_input, unsigned char* d_output, unsigned char* d_temp,
                           float* d_kernel) {
  if (d_input != nullptr)
    cudaFree(d_input);
  if (d_output != nullptr)
    cudaFree(d_output);
  if (d_temp != nullptr)
    cudaFree(d_temp);
  if (d_kernel != nullptr)
    cudaFree(d_kernel);
}

static cudaError_t allocate_memory(int width, int height, int channels, int kernel_size,
                                   unsigned char** d_input, unsigned char** d_output,
                                   unsigned char** d_temp, float** d_kernel) {
  size_t data_size = width * height * channels;
  cudaError_t error;

  error = cudaMalloc(d_input, data_size);
  if (error != cudaSuccess)
    return error;

  error = cudaMalloc(d_output, data_size);
  if (error != cudaSuccess) {
    cudaFree(*d_input);
    return error;
  }

  error = cudaMalloc(d_temp, data_size);
  if (error != cudaSuccess) {
    cleanup_memory(*d_input, *d_output, nullptr, nullptr);
    return error;
  }

  error = cudaMalloc(d_kernel, kernel_size * sizeof(float));
  if (error != cudaSuccess) {
    cleanup_memory(*d_input, *d_output, *d_temp, nullptr);
    return error;
  }

  return cudaSuccess;
}

extern "C" cudaError_t cuda_apply_gaussian_blur_1d_horizontal(const unsigned char* input,
                                                              unsigned char* output, int width,
                                                              int height, int channels,
                                                              const float* kernel, int kernel_size,
                                                              int border_mode) {
  unsigned char* d_input = nullptr;
  unsigned char* d_output = nullptr;
  float* d_kernel = nullptr;

  cudaError_t error = allocate_memory(width, height, channels, kernel_size, &d_input, &d_output,
                                      nullptr, &d_kernel);
  if (error != cudaSuccess)
    return error;

  size_t data_size = width * height * channels;
  error = cudaMemcpy(d_input, input, data_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, nullptr, d_kernel);
    return error;
  }

  error = cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, nullptr, d_kernel);
    return error;
  }

  BorderMode kernel_border_mode = static_cast<BorderMode>(border_mode);
  dim3 block_size(256);
  dim3 grid_size((width + block_size.x - 1) / block_size.x);
  ApplyHorizontalBlurKernel<<<grid_size, block_size>>>(d_input, d_output, width, height, channels,
                                                       d_kernel, kernel_size, kernel_border_mode);

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, nullptr, d_kernel);
    return error;
  }

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, nullptr, d_kernel);
    return error;
  }

  error = cudaMemcpy(output, d_output, data_size, cudaMemcpyDeviceToHost);
  cleanup_memory(d_input, d_output, nullptr, d_kernel);
  return error;
}

extern "C" cudaError_t cuda_apply_gaussian_blur_1d_vertical(const unsigned char* input,
                                                            unsigned char* output, int width,
                                                            int height, int channels,
                                                            const float* kernel, int kernel_size,
                                                            int border_mode) {
  unsigned char* d_input = nullptr;
  unsigned char* d_output = nullptr;
  float* d_kernel = nullptr;

  cudaError_t error = allocate_memory(width, height, channels, kernel_size, &d_input, &d_output,
                                      nullptr, &d_kernel);
  if (error != cudaSuccess)
    return error;

  size_t data_size = width * height * channels;
  error = cudaMemcpy(d_input, input, data_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, nullptr, d_kernel);
    return error;
  }

  error = cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, nullptr, d_kernel);
    return error;
  }

  BorderMode kernel_border_mode = static_cast<BorderMode>(border_mode);
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  ApplyVerticalBlurKernel<<<grid_size, block_size>>>(d_input, d_output, width, height, channels,
                                                     d_kernel, kernel_size, kernel_border_mode);

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, nullptr, d_kernel);
    return error;
  }

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, nullptr, d_kernel);
    return error;
  }

  error = cudaMemcpy(output, d_output, data_size, cudaMemcpyDeviceToHost);
  cleanup_memory(d_input, d_output, nullptr, d_kernel);
  return error;
}

extern "C" cudaError_t cuda_apply_gaussian_blur_separable(const unsigned char* input,
                                                          unsigned char* output, int width,
                                                          int height, int channels,
                                                          const float* kernel, int kernel_size,
                                                          int border_mode) {
  unsigned char* d_input = nullptr;
  unsigned char* d_output = nullptr;
  unsigned char* d_temp = nullptr;
  float* d_kernel = nullptr;

  cudaError_t error = allocate_memory(width, height, channels, kernel_size, &d_input, &d_output,
                                      &d_temp, &d_kernel);
  if (error != cudaSuccess)
    return error;

  size_t data_size = width * height * channels;
  error = cudaMemcpy(d_input, input, data_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, d_temp, d_kernel);
    return error;
  }

  error = cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, d_temp, d_kernel);
    return error;
  }

  BorderMode kernel_border_mode = static_cast<BorderMode>(border_mode);

  dim3 block_size_horizontal(256);
  dim3 grid_size_horizontal((width + block_size_horizontal.x - 1) / block_size_horizontal.x);
  ApplyHorizontalBlurKernel<<<grid_size_horizontal, block_size_horizontal>>>(
      d_input, d_temp, width, height, channels, d_kernel, kernel_size, kernel_border_mode);

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, d_temp, d_kernel);
    return error;
  }

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, d_temp, d_kernel);
    return error;
  }

  dim3 block_size_vertical(16, 16);
  dim3 grid_size_vertical((width + block_size_vertical.x - 1) / block_size_vertical.x,
                          (height + block_size_vertical.y - 1) / block_size_vertical.y);
  ApplyVerticalBlurKernel<<<grid_size_vertical, block_size_vertical>>>(
      d_temp, d_output, width, height, channels, d_kernel, kernel_size, kernel_border_mode);

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, d_temp, d_kernel);
    return error;
  }

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cleanup_memory(d_input, d_output, d_temp, d_kernel);
    return error;
  }

  error = cudaMemcpy(output, d_output, data_size, cudaMemcpyDeviceToHost);
  cleanup_memory(d_input, d_output, d_temp, d_kernel);
  return error;
}

}  // namespace jrb::infrastructure::cuda

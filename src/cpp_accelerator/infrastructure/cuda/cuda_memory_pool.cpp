#include "src/cpp_accelerator/infrastructure/cuda/cuda_memory_pool.h"

#include <cuda_runtime.h>

#include <iostream>

namespace jrb::infrastructure::cuda {

CudaMemoryPool::~CudaMemoryPool() {
  Clear();
}

void* CudaMemoryPool::Allocate(std::size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = free_buffers_.find(size);
  if (it != free_buffers_.end() && !it->second.empty()) {
    void* ptr = it->second.back();
    it->second.pop_back();
    allocated_sizes_[ptr] = size;
    return ptr;
  }

  void* ptr = nullptr;
  cudaError_t error = cudaMalloc(&ptr, size);
  if (error != cudaSuccess) {
    std::cerr << "[CudaMemoryPool] Failed to allocate " << size << " bytes: " << cudaGetErrorString(error) << std::endl;
    return nullptr;
  }

  allocated_sizes_[ptr] = size;
  total_allocated_bytes_ += size;
  return ptr;
}

void CudaMemoryPool::Deallocate(void* ptr, std::size_t size) {
  if (!ptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocated_sizes_.find(ptr);
  if (it == allocated_sizes_.end()) {
    std::cerr << "[CudaMemoryPool] Attempting to deallocate unknown pointer" << std::endl;
    return;
  }

  allocated_sizes_.erase(it);
  free_buffers_[size].push_back(ptr);
}

void CudaMemoryPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto& [size, buffers] : free_buffers_) {
    for (void* ptr : buffers) {
      if (ptr) {
        cudaFree(ptr);
      }
    }
  }
  free_buffers_.clear();

  for (auto& [ptr, size] : allocated_sizes_) {
    if (ptr) {
      cudaFree(ptr);
    }
  }
  allocated_sizes_.clear();

  total_allocated_bytes_ = 0;
}

std::size_t CudaMemoryPool::GetPoolSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t count = 0;
  for (const auto& [size, buffers] : free_buffers_) {
    count += buffers.size();
  }
  return count;
}

std::size_t CudaMemoryPool::GetTotalAllocatedBytes() const {
  return total_allocated_bytes_;
}

CudaMemoryPool& GetThreadLocalMemoryPool() {
  thread_local CudaMemoryPool pool;
  return pool;
}

}  // namespace jrb::infrastructure::cuda

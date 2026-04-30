#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace jrb::adapters::compute::cuda {

class CudaMemoryPool {
public:
  CudaMemoryPool() = default;
  ~CudaMemoryPool();

  void* Allocate(std::size_t size);
  void Deallocate(void* ptr, std::size_t size);

  void Clear();

  std::size_t GetPoolSize() const;
  std::size_t GetTotalAllocatedBytes() const;

private:
  CudaMemoryPool(const CudaMemoryPool&) = delete;
  CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

  mutable std::mutex mutex_;
  std::unordered_map<std::size_t, std::deque<void*>> free_buffers_;
  std::unordered_map<void*, std::size_t> allocated_sizes_;
  std::size_t total_allocated_bytes_ = 0;
};

// Thread-local accessor for the current thread's memory pool
// Each WebRTC processing thread gets its own isolated pool
CudaMemoryPool& GetThreadLocalMemoryPool();

}  // namespace jrb::adapters::compute::cuda

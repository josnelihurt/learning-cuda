#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace jrb::application::pipeline {

class BufferPool {
public:
  explicit BufferPool(size_t initial_capacity = 4);
  ~BufferPool() = default;

  std::vector<unsigned char>* Acquire(size_t size);
  void Release(std::vector<unsigned char>* buffer);
  void Clear();

  size_t GetAvailableCount() const;
  size_t GetTotalCount() const;

private:
  std::vector<std::unique_ptr<std::vector<unsigned char>>> available_buffers_;
  std::vector<std::unique_ptr<std::vector<unsigned char>>> in_use_buffers_;
  size_t total_created_;
};

}  // namespace jrb::application::pipeline

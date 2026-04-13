#include "cpp_accelerator/application/pipeline/buffer_pool.h"

#include <algorithm>

namespace jrb::application::pipeline {

BufferPool::BufferPool(size_t initial_capacity) : total_created_(0) {
  available_buffers_.reserve(initial_capacity);
  in_use_buffers_.reserve(initial_capacity);
}

std::vector<unsigned char>* BufferPool::Acquire(size_t size) {
  auto it = std::find_if(available_buffers_.begin(), available_buffers_.end(),
                         [size](const std::unique_ptr<std::vector<unsigned char>>& buffer) {
                           return buffer && buffer->size() >= size;
                         });

  if (it != available_buffers_.end()) {
    std::unique_ptr<std::vector<unsigned char>> buffer = std::move(*it);
    available_buffers_.erase(it);
    if (buffer->size() < size) {
      buffer->resize(size);
    }
    std::vector<unsigned char>* ptr = buffer.get();
    in_use_buffers_.push_back(std::move(buffer));
    return ptr;
  }

  auto new_buffer = std::make_unique<std::vector<unsigned char>>(size);
  std::vector<unsigned char>* ptr = new_buffer.get();
  in_use_buffers_.push_back(std::move(new_buffer));
  total_created_++;
  return ptr;
}

void BufferPool::Release(std::vector<unsigned char>* buffer) {
  auto it = std::find_if(
      in_use_buffers_.begin(), in_use_buffers_.end(),
      [buffer](const std::unique_ptr<std::vector<unsigned char>>& b) { return b.get() == buffer; });
  if (it != in_use_buffers_.end()) {
    std::unique_ptr<std::vector<unsigned char>> released = std::move(*it);
    in_use_buffers_.erase(it);
    available_buffers_.push_back(std::move(released));
  }
}

void BufferPool::Clear() {
  available_buffers_.clear();
  in_use_buffers_.clear();
  total_created_ = 0;
}

size_t BufferPool::GetAvailableCount() const {
  return available_buffers_.size();
}

size_t BufferPool::GetTotalCount() const {
  return total_created_;
}

}  // namespace jrb::application::pipeline

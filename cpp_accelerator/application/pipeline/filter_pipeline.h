#pragma once

#include <memory>
#include <vector>

#include "cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::application::pipeline {

class BufferPool;

class FilterPipeline {
public:
  explicit FilterPipeline(std::unique_ptr<BufferPool> buffer_pool = nullptr);
  ~FilterPipeline() = default;

  void AddFilter(std::unique_ptr<jrb::domain::interfaces::IFilter> filter);
  bool Apply(const jrb::domain::interfaces::ImageBuffer& input,
             jrb::domain::interfaces::ImageBufferMut& output);

  size_t GetFilterCount() const;
  void Clear();

private:
  std::vector<std::unique_ptr<jrb::domain::interfaces::IFilter>> filters_;
  std::unique_ptr<BufferPool> buffer_pool_;
};

}  // namespace jrb::application::pipeline

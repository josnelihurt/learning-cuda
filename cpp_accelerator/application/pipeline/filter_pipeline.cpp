#include "cpp_accelerator/application/pipeline/filter_pipeline.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "cpp_accelerator/application/pipeline/buffer_pool.h"
#include "cpp_accelerator/domain/interfaces/filters/i_filter.h"
#include "cpp_accelerator/domain/interfaces/image_buffer.h"

namespace jrb::application::pipeline {

using jrb::domain::interfaces::FilterContext;
using jrb::domain::interfaces::IFilter;
using jrb::domain::interfaces::ImageBuffer;
using jrb::domain::interfaces::ImageBufferMut;

FilterPipeline::FilterPipeline(std::unique_ptr<BufferPool> buffer_pool)
    : buffer_pool_(buffer_pool ? std::move(buffer_pool) : std::make_unique<BufferPool>()) {}

void FilterPipeline::AddFilter(std::unique_ptr<IFilter> filter) {
  if (filter) {
    filters_.push_back(std::move(filter));
  }
}

size_t FilterPipeline::GetFilterCount() const {
  return filters_.size();
}

void FilterPipeline::Clear() {
  filters_.clear();
}

bool FilterPipeline::Apply(const ImageBuffer& input, ImageBufferMut& output) {
  if (filters_.empty()) {
    return false;
  }

  if (!input.IsValid() || !output.IsValid()) {
    return false;
  }

  int current_width = input.width;
  int current_height = input.height;
  int current_channels = input.channels;
  const unsigned char* current_input = input.data;

  std::vector<std::vector<unsigned char>*> intermediate_buffers;

  for (size_t i = 0; i < filters_.size(); ++i) {
    bool is_last = (i == filters_.size() - 1);
    IFilter* filter = filters_[i].get();

    int next_channels = current_channels;
    if (filter->GetType() == jrb::domain::interfaces::FilterType::GRAYSCALE) {
      next_channels = 1;
    }

    size_t required_size = static_cast<size_t>(current_width) * current_height * next_channels;

    std::vector<unsigned char>* output_buffer = nullptr;
    if (is_last) {
      if (output.width != current_width || output.height != current_height) {
        return false;
      }
    } else {
      output_buffer = buffer_pool_->Acquire(required_size);
      intermediate_buffers.push_back(output_buffer);
    }

    unsigned char* output_ptr = is_last ? output.data : output_buffer->data();

    int output_channels = is_last ? output.channels : next_channels;
    FilterContext context(current_input, output_ptr, current_width, current_height,
                          current_channels);

    jrb::domain::interfaces::ImageBufferMut output_buffer_mut(output_ptr, current_width,
                                                              current_height, output_channels);
    context.output = output_buffer_mut;

    if (!filter->Apply(context)) {
      for (auto* buf : intermediate_buffers) {
        buffer_pool_->Release(buf);
      }
      return false;
    }

    if (!is_last) {
      current_input = output_ptr;
      current_channels = next_channels;
    }
  }

  for (auto* buf : intermediate_buffers) {
    buffer_pool_->Release(buf);
  }

  return true;
}

}  // namespace jrb::application::pipeline

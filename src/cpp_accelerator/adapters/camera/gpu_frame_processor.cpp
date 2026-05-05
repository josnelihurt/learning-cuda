#include "src/cpp_accelerator/adapters/camera/gpu_frame_processor.h"

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <string_view>

#include <spdlog/spdlog.h>

#include "src/cpp_accelerator/adapters/camera/nvbuf_cuda_utils.h"
#include "src/cpp_accelerator/adapters/compute/cuda/kernels/nv12_utils_kernel.h"

namespace jrb::adapters::camera {
constexpr std::string_view kLogPrefix = "[GpuFrameProcessor]";

struct GpuFrameProcessor::Impl {
  std::atomic<bool> running{false};

  std::mutex rgb_cb_mutex;
  RgbCallback rgb_cb;

  int configured_width = 0;
  int configured_height = 0;

  // Scratch buffers used only when an RgbCallback is active.
  uint8_t* d_y_in = nullptr;   // pitch x height
  uint8_t* d_uv_in = nullptr;  // pitch x (height/2)
  uint8_t* d_rgba = nullptr;   // width x height x 4

  int alloc_width = 0;
  int alloc_height = 0;
  int alloc_pitch = 0;

  std::vector<uint8_t> h_rgba;

  bool EnsureScratch(int w, int h, int p) {
    if (d_rgba && w == alloc_width && h == alloc_height && p == alloc_pitch) {
      return true;
    }
    if (d_rgba) {
      spdlog::warn(
          "[GpuFrameProcessor] Frame layout changed {}x{} pitch={} -> {}x{} pitch={}; "
          "reallocating scratch",
          alloc_width, alloc_height, alloc_pitch, w, h, p);
      FreeScratch();
    }
    const size_t in_y_bytes = static_cast<size_t>(p) * h;
    const size_t in_uv_bytes = static_cast<size_t>(p) * (h / 2);
    const size_t rgba_bytes = static_cast<size_t>(w) * h * 4;

    if (cudaMalloc(&d_y_in, in_y_bytes) != cudaSuccess ||
        cudaMalloc(&d_uv_in, in_uv_bytes) != cudaSuccess ||
        cudaMalloc(&d_rgba, rgba_bytes) != cudaSuccess) {
      spdlog::error("{} cudaMalloc failed for scratch buffers", kLogPrefix);
      FreeScratch();
      return false;
    }
    h_rgba.resize(rgba_bytes);
    alloc_width = w;
    alloc_height = h;
    alloc_pitch = p;
    return true;
  }

  void FreeScratch() {
    if (d_y_in) {
      cudaFree(d_y_in);
      d_y_in = nullptr;
    }
    if (d_uv_in) {
      cudaFree(d_uv_in);
      d_uv_in = nullptr;
    }
    if (d_rgba) {
      cudaFree(d_rgba);
      d_rgba = nullptr;
    }
    h_rgba.clear();
    h_rgba.shrink_to_fit();
    alloc_width = alloc_height = alloc_pitch = 0;
  }
};

GpuFrameProcessor::GpuFrameProcessor() : impl_(std::make_unique<Impl>()) {}
GpuFrameProcessor::~GpuFrameProcessor() {
  Stop();
}

bool GpuFrameProcessor::Start(int width, int height, std::string* /*error_message*/) {
  impl_->configured_width = width;
  impl_->configured_height = height;
  impl_->running = true;
  return true;
}

void GpuFrameProcessor::Stop() {
  impl_->running = false;
  {
    std::lock_guard<std::mutex> lk(impl_->rgb_cb_mutex);
    impl_->rgb_cb = nullptr;
  }
  impl_->FreeScratch();
  impl_->configured_width = 0;
  impl_->configured_height = 0;
}

bool GpuFrameProcessor::IsRunning() const {
  return impl_->running.load();
}

void GpuFrameProcessor::SetRgbCallback(RgbCallback cb) {
  std::lock_guard<std::mutex> lk(impl_->rgb_cb_mutex);
  impl_->rgb_cb = std::move(cb);
}

void GpuFrameProcessor::Process(GstBuffer* nvmm_buf, uint32_t /*rtp_ts*/) {
  if (!impl_->running.load())
    return;

  // Snapshot the callback so we can drop the lock before doing CUDA work.
  RgbCallback cb_snapshot;
  {
    std::lock_guard<std::mutex> lk(impl_->rgb_cb_mutex);
    cb_snapshot = impl_->rgb_cb;
  }
  if (!cb_snapshot) {
    // Nothing consumes RGBA right now; skip the entire pipeline so the
    // streaming path pays no GPU/CPU tax.
    return;
  }

  GstMapInfo map_info{};
  NvmmFrame frame{};
  if (!MapNvmmBuffer(nvmm_buf, &map_info, &frame)) {
    spdlog::warn("{} MapNvmmBuffer failed; dropping frame", kLogPrefix);
    return;
  }

  const int w = frame.width;
  const int h = frame.height;
  const int pitch = frame.pitch;

  if (!impl_->EnsureScratch(w, h, pitch)) {
    UnmapNvmmBuffer(nvmm_buf, &map_info);
    return;
  }

  const size_t in_y_bytes = static_cast<size_t>(pitch) * h;
  const size_t in_uv_bytes = static_cast<size_t>(pitch) * (h / 2);
  if (cudaMemcpy(impl_->d_y_in, frame.y_ptr, in_y_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(impl_->d_uv_in, frame.uv_ptr, in_uv_bytes, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    spdlog::error("{} NV12 H->D copy failed", kLogPrefix);
    UnmapNvmmBuffer(nvmm_buf, &map_info);
    return;
  }

  const cudaError_t conv_err =
      cuda_nv12_to_rgba_device(impl_->d_y_in, impl_->d_uv_in, pitch, impl_->d_rgba, w, h);
  if (conv_err != cudaSuccess) {
    spdlog::error("{} cuda_nv12_to_rgba_device: {}", kLogPrefix, cudaGetErrorString(conv_err));
    UnmapNvmmBuffer(nvmm_buf, &map_info);
    return;
  }

  const size_t rgba_bytes = static_cast<size_t>(w) * h * 4;
  if (cudaMemcpy(impl_->h_rgba.data(), impl_->d_rgba, rgba_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    spdlog::error("{} RGBA D->H copy failed", kLogPrefix);
    UnmapNvmmBuffer(nvmm_buf, &map_info);
    return;
  }

  UnmapNvmmBuffer(nvmm_buf, &map_info);

  try {
    cb_snapshot(impl_->h_rgba, w, h);
  } catch (const std::exception& e) {
    spdlog::warn("{} RgbCallback threw: {}", kLogPrefix, e.what());
  }
}

}  // namespace jrb::adapters::camera

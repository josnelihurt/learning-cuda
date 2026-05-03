#include "src/cpp_accelerator/adapters/camera/gpu_frame_processor.h"

#include <cuda_runtime.h>
#include <mutex>

#include <spdlog/spdlog.h>

#include "src/cpp_accelerator/adapters/camera/encode_pipeline.h"
#include "src/cpp_accelerator/adapters/camera/nvbuf_cuda_utils.h"
#include "src/cpp_accelerator/adapters/compute/cuda/kernels/nv12_utils_kernel.h"

namespace jrb::adapters::camera {

struct GpuFrameProcessor::Impl {
  std::unique_ptr<EncodePipeline> encode_pipeline;

  std::mutex rgb_cb_mutex;
  RgbCallback rgb_cb;

  int width  = 0;
  int height = 0;

  // Scratch CUDA device buffers.  Allocated once on first Process() call.
  uint8_t* d_rgba     = nullptr;  // width × height × 4
  uint8_t* d_y_out    = nullptr;  // width × height      (processed Y plane)
  uint8_t* d_uv_out   = nullptr;  // width × (height/2)  (processed UV plane)

  // Host staging buffers.
  std::vector<uint8_t> h_nv12;   // for EncodePipeline::PushFrame
  std::vector<uint8_t> h_rgba;   // for RgbCallback (allocated lazily)

  bool AllocScratch(int w, int h) {
    if (d_rgba) {
      if (w != width || h != height) {
        spdlog::error("[GpuFrameProcessor] Frame dimensions changed {}x{} -> {}x{}; "
                      "reallocating scratch buffers", width, height, w, h);
        FreeScratch();
      } else {
        return true;
      }
    }
    const size_t rgba_bytes = static_cast<size_t>(w) * h * 4;
    const size_t y_bytes    = static_cast<size_t>(w) * h;
    const size_t uv_bytes   = static_cast<size_t>(w) * (h / 2);

    if (cudaMalloc(&d_rgba, rgba_bytes) != cudaSuccess ||
        cudaMalloc(&d_y_out, y_bytes) != cudaSuccess ||
        cudaMalloc(&d_uv_out, uv_bytes) != cudaSuccess) {
      spdlog::error("[GpuFrameProcessor] cudaMalloc failed for scratch buffers");
      FreeScratch();
      return false;
    }

    h_nv12.resize(y_bytes + uv_bytes);
    return true;
  }

  void FreeScratch() {
    if (d_rgba)   { cudaFree(d_rgba);   d_rgba   = nullptr; }
    if (d_y_out)  { cudaFree(d_y_out);  d_y_out  = nullptr; }
    if (d_uv_out) { cudaFree(d_uv_out); d_uv_out = nullptr; }
    h_nv12.clear();
    h_rgba.clear();
  }
};

GpuFrameProcessor::GpuFrameProcessor() : impl_(std::make_unique<Impl>()) {}
GpuFrameProcessor::~GpuFrameProcessor() { Stop(); }

bool GpuFrameProcessor::Start(int width, int height, int fps, FrameCallback h264_cb,
                               std::string* error_message) {
  impl_->width  = width;
  impl_->height = height;

  impl_->encode_pipeline = std::make_unique<EncodePipeline>();
  if (!impl_->encode_pipeline->Start(width, height, fps, std::move(h264_cb), error_message)) {
    impl_->encode_pipeline.reset();
    return false;
  }
  return true;
}

void GpuFrameProcessor::Stop() {
  if (impl_->encode_pipeline) {
    impl_->encode_pipeline->Stop();
    impl_->encode_pipeline.reset();
  }
  impl_->FreeScratch();
  impl_->width  = 0;
  impl_->height = 0;
}

bool GpuFrameProcessor::IsRunning() const {
  return impl_->encode_pipeline && impl_->encode_pipeline->IsRunning();
}

void GpuFrameProcessor::SetRgbCallback(RgbCallback cb) {
  std::lock_guard<std::mutex> lk(impl_->rgb_cb_mutex);
  impl_->rgb_cb = std::move(cb);
}

void GpuFrameProcessor::Process(GstBuffer* nvmm_buf, uint32_t rtp_ts) {
  if (!impl_->encode_pipeline || !impl_->encode_pipeline->IsRunning()) return;

  // ── 1. Map NVMM → CUDA device pointers ──────────────────────────────────
  GstMapInfo map_info{};
  NvmmFrame frame{};
  if (!MapNvmmBuffer(nvmm_buf, &map_info, &frame)) {
    spdlog::warn("[GpuFrameProcessor] MapNvmmBuffer failed — dropping frame");
    return;
  }

  const int w     = frame.width;
  const int h     = frame.height;
  const int pitch = frame.pitch;

  // ── 2. Ensure scratch buffers exist ──────────────────────────────────────
  if (!impl_->AllocScratch(w, h)) {
    UnmapNvmmBuffer(nvmm_buf, &map_info);
    return;
  }

  // ── 3. NV12 → RGBA (device) ──────────────────────────────────────────────
  cudaError_t cuda_err =
      cuda_nv12_to_rgba_device(frame.y_ptr, frame.uv_ptr, pitch, impl_->d_rgba, w, h);
  if (cuda_err != cudaSuccess) {
    spdlog::error("[GpuFrameProcessor] cuda_nv12_to_rgba_device: {}",
                  cudaGetErrorString(cuda_err));
    UnmapNvmmBuffer(nvmm_buf, &map_info);
    return;
  }

  // ── 4. Optional GPU filter kernels go here in the future ─────────────────
  //      (grayscale / blur device-ptr overloads can be inserted here)

  // ── 5. RGBA → NV12 (device), writing to scratch Y/UV planes ─────────────
  //    Pitch for scratch output equals width (dense layout for EncodePipeline).
  cuda_err =
      cuda_rgba_to_nv12_device(impl_->d_rgba, impl_->d_y_out, impl_->d_uv_out, w, w, h);
  if (cuda_err != cudaSuccess) {
    spdlog::error("[GpuFrameProcessor] cuda_rgba_to_nv12_device: {}",
                  cudaGetErrorString(cuda_err));
    UnmapNvmmBuffer(nvmm_buf, &map_info);
    return;
  }

  // ── 6. Download processed NV12 to host → EncodePipeline ─────────────────
  {
    const size_t y_bytes  = static_cast<size_t>(w) * h;
    const size_t uv_bytes = static_cast<size_t>(w) * (h / 2);

    if (cudaMemcpy(impl_->h_nv12.data(), impl_->d_y_out, y_bytes, cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(impl_->h_nv12.data() + y_bytes, impl_->d_uv_out, uv_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      spdlog::error("[GpuFrameProcessor] NV12 D→H copy failed");
      UnmapNvmmBuffer(nvmm_buf, &map_info);
      return;
    }
    impl_->encode_pipeline->PushFrame(impl_->h_nv12.data(), w, h, rtp_ts);
  }

  // ── 7. Optional RGBA download for RgbCallback (YOLO / BirdWatcher) ───────
  {
    std::lock_guard<std::mutex> lk(impl_->rgb_cb_mutex);
    if (impl_->rgb_cb) {
      const size_t rgba_bytes = static_cast<size_t>(w) * h * 4;
      impl_->h_rgba.resize(rgba_bytes);
      if (cudaMemcpy(impl_->h_rgba.data(), impl_->d_rgba, rgba_bytes,
                     cudaMemcpyDeviceToHost) == cudaSuccess) {
        try {
          impl_->rgb_cb(impl_->h_rgba, w, h);
        } catch (const std::exception& e) {
          spdlog::warn("[GpuFrameProcessor] RgbCallback threw: {}", e.what());
        }
      } else {
        spdlog::error("[GpuFrameProcessor] RGBA D→H copy failed");
      }
    }
  }

  // ── 8. Release NVMM mapping ───────────────────────────────────────────────
  UnmapNvmmBuffer(nvmm_buf, &map_info);
}

}  // namespace jrb::adapters::camera

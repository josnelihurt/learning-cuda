// GPU-path wiring for BirdWatcher on Jetson / NvidiaArgusBackend.
// Compiled only when --config=nvidia-argus-camera is set (select() in BUILD).

#include "src/cpp_accelerator/application/bird_watch/bird_watcher.h"

#include <spdlog/spdlog.h>

#include "src/cpp_accelerator/adapters/camera/gpu_frame_processor.h"

namespace jrb::application::bird_watch {

void BirdWatcher::ConnectGpuPath() {
  gpu_processor_ = camera_hub_->GetGpuFrameProcessor(config_.camera_sensor_id);
  if (gpu_processor_) {
    spdlog::info("[BirdWatcher] GPU direct path active — bypassing H.264 decode");
    DestroyDecoder();
    gpu_processor_->SetRgbCallback(
        [this](const std::vector<uint8_t>& rgba, int w, int h) { OnRgbaFrame(rgba, w, h); });
  } else {
    spdlog::info("[BirdWatcher] No GpuFrameProcessor found — using H.264 decode path");
  }
}

void BirdWatcher::DisconnectGpuPath() {
  if (gpu_processor_) {
    gpu_processor_->SetRgbCallback(nullptr);
    gpu_processor_ = nullptr;
  }
}

}  // namespace jrb::application::bird_watch

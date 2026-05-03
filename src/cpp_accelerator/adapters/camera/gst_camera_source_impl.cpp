#include "src/cpp_accelerator/adapters/camera/gst_camera_source_impl.h"

#include "src/cpp_accelerator/adapters/camera/backends/camera_backend.h"
#include "src/cpp_accelerator/adapters/camera/backends/stub_backend.h"

#ifdef CAMERA_BACKEND_V4L2_ENABLED
#include "src/cpp_accelerator/adapters/camera/backends/v4l2_backend.h"
#endif

#ifdef CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED
#include "src/cpp_accelerator/adapters/camera/backends/nvidia_argus_backend.h"
#endif

#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

GstCameraSourceImpl::GstCameraSourceImpl() {
  // Initialize backends in priority order for streaming.
#ifdef CAMERA_BACKEND_V4L2_ENABLED
  backends_.push_back(std::make_unique<V4L2Backend>());
#endif

#ifdef CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED
  backends_.push_back(std::make_unique<NvidiaArgusBackend>());
#endif

  // Stub is always available as fallback
  backends_.push_back(std::make_unique<StubBackend>());
}

GstCameraSourceImpl::~GstCameraSourceImpl() {
  Stop();
}

void GstCameraSourceImpl::SetFrameCallback(FrameCallback cb) {
  if (active_backend_) {
    active_backend_->SetFrameCallback(std::move(cb));
  }
  // Store for when a backend becomes active
  for (auto& backend : backends_) {
    if (backend && backend != active_backend_) {
      backend->SetFrameCallback(cb);
    }
  }
}

bool GstCameraSourceImpl::Start(int sensor_id, int width, int height, int fps,
                               std::string* error_message) {
  if (IsRunning()) {
    if (error_message) *error_message = "Already streaming from another sensor";
    spdlog::warn("[GstCameraSourceImpl] Already streaming");
    return false;
  }

  // Try each backend in priority order until one succeeds
  for (auto& backend : backends_) {
    if (!backend->IsAvailable()) {
      spdlog::debug("[GstCameraSourceImpl] Backend {} not available", backend->GetBackendName());
      continue;
    }

    spdlog::info("[GstCameraSourceImpl] Trying backend: {}", backend->GetBackendName());
    std::string backend_error;
    if (backend->Start(sensor_id, width, height, fps, &backend_error)) {
      active_backend_ = std::move(backend);
      spdlog::info("[GstCameraSourceImpl] Successfully started with {} backend",
                   active_backend_->GetBackendName());
      return true;
    }
    spdlog::warn("[GstCameraSourceImpl] Backend {} failed: {}", 
                 backend->GetBackendName(), backend_error);
  }

  // All backends failed
  const std::string msg =
      "All camera backends failed to start streaming for sensor_id=" + std::to_string(sensor_id);
  if (error_message) *error_message = msg;
  spdlog::error("[GstCameraSourceImpl] {}", msg);
  return false;
}

void GstCameraSourceImpl::Stop() {
  if (active_backend_) {
    active_backend_->Stop();
    active_backend_ = nullptr;
  }
}

bool GstCameraSourceImpl::IsRunning() const {
  return active_backend_ && active_backend_->IsRunning();
}

rtc::binary GstCameraSourceImpl::GrabStillFrame(int* out_width, int* out_height) {
  if (!active_backend_) return {};
  return active_backend_->GrabStillFrame(out_width, out_height);
}

}  // namespace jrb::adapters::camera

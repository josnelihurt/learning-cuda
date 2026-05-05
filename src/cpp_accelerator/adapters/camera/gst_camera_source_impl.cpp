#include "src/cpp_accelerator/adapters/camera/gst_camera_source_impl.h"

#include "src/cpp_accelerator/adapters/camera/backends/camera_backend.h"
#include "src/cpp_accelerator/adapters/camera/backends/stub_backend.h"

#include <spdlog/spdlog.h>
#include <string_view>

namespace jrb::adapters::camera {
constexpr std::string_view kLogPrefix = "[GstCameraSourceImpl]";

GstCameraSourceImpl::GstCameraSourceImpl() {
  RegisterV4L2Backend();
  RegisterArgusBackend();
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
    if (error_message)
      *error_message = "Already streaming from another sensor";
    spdlog::warn("{} Already streaming", kLogPrefix);
    return false;
  }

  // Try each backend in priority order until one succeeds
  for (auto& backend : backends_) {
    if (!backend->IsAvailable()) {
      spdlog::debug("{} Backend {} not available", kLogPrefix, backend->GetBackendName());
      continue;
    }

    spdlog::info("{} Trying backend: {}", kLogPrefix, backend->GetBackendName());
    std::string backend_error;
    if (backend->Start(sensor_id, width, height, fps, &backend_error)) {
      active_backend_ = std::move(backend);
      spdlog::info("{} Successfully started with {} backend", kLogPrefix,
                   active_backend_->GetBackendName());
      return true;
    }
    spdlog::warn("{} Backend {} failed: {}", kLogPrefix, backend->GetBackendName(), backend_error);
  }

  // All backends failed
  const std::string msg =
      "All camera backends failed to start streaming for sensor_id=" + std::to_string(sensor_id);
  if (error_message)
    *error_message = msg;
  spdlog::error("{} {}", kLogPrefix, msg);
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
  if (!active_backend_)
    return {};
  return active_backend_->GrabStillFrame(out_width, out_height);
}

}  // namespace jrb::adapters::camera

#include "src/cpp_accelerator/adapters/camera/backends/stub_backend.h"

#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

struct StubBackend::Impl {
  bool running = false;
};

StubBackend::StubBackend() : impl_(std::make_unique<Impl>()) {}

StubBackend::~StubBackend() = default;

bool StubBackend::IsAvailable() const {
  return true;
}

std::vector<cuda_learning::RemoteCameraInfo> StubBackend::DetectCameras(
    const std::vector<int>& /*sensor_ids*/) {
  return {};
}

void StubBackend::SetFrameCallback(FrameCallback /*cb*/) {}

bool StubBackend::Start(int sensor_id, int /*width*/, int /*height*/,
                        int /*fps*/, std::string* error_message) {
  spdlog::warn(
      "[StubBackend] No video from Stub fallback (sensor_id={}); "
      "hardware camera backends failed or were unavailable — see prior logs",
      sensor_id);
  if (error_message) {
    *error_message =
        "Stub backend cannot stream video; Argus/V4L2 backends did not start.";
  }
  return false;
}

void StubBackend::Stop() {}

bool StubBackend::IsRunning() const {
  return impl_->running;
}

std::string StubBackend::GetBackendName() const {
  return "Stub";
}

}  // namespace jrb::adapters::camera

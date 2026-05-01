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
      "[StubBackend] Camera streaming not available. "
      "No camera backends are enabled. "
      "Enable with: --//bazel/flags:v4l2_camera=true or "
      "--//bazel/flags:nvidia_argus_camera=true (sensor_id={})",
      sensor_id);
  if (error_message) {
    *error_message = "No camera backends enabled. Enable v4l2_camera or nvidia_argus_camera flags.";
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

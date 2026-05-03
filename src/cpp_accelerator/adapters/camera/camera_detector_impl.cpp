#include "src/cpp_accelerator/adapters/camera/camera_detector_impl.h"

#include "src/cpp_accelerator/adapters/camera/backends/camera_backend.h"
#include "src/cpp_accelerator/adapters/camera/backends/stub_backend.h"

#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

CameraDetectorImpl::CameraDetectorImpl() {
  RegisterV4L2Backend();
  RegisterArgusBackend();
  backends_.push_back(std::make_unique<StubBackend>());

  std::string backend_list;
  for (size_t i = 0; i < backends_.size(); ++i) {
    if (i > 0) {
      backend_list += ", ";
    }
    backend_list += backends_[i]->GetBackendName();
  }
  spdlog::info("[CameraDetector] Backends initialized: {}", backend_list);
}

CameraDetectorImpl::~CameraDetectorImpl() = default;

std::vector<cuda_learning::RemoteCameraInfo> CameraDetectorImpl::DetectCameras(
    const std::vector<int>& sensor_ids) {
  std::vector<cuda_learning::RemoteCameraInfo> all_cameras;

  // Try each backend and aggregate results.
  for (auto& backend : backends_) {
    if (!backend->IsAvailable()) {
      spdlog::debug("[CameraDetector] Backend {} not available", backend->GetBackendName());
      continue;
    }

    spdlog::info("[CameraDetector] Trying backend: {}", backend->GetBackendName());
    auto cameras = backend->DetectCameras(sensor_ids);
    spdlog::info("[CameraDetector] Backend {} detected {} camera(s)",
                 backend->GetBackendName(), cameras.size());
    all_cameras.insert(all_cameras.end(),
                       std::make_move_iterator(cameras.begin()),
                       std::make_move_iterator(cameras.end()));
  }

  if (all_cameras.empty()) {
    spdlog::info("[CameraDetector] No cameras detected by any backend");
  } else {
    spdlog::info("[CameraDetector] Detected {} total cameras", all_cameras.size());
  }

  return all_cameras;
}

}  // namespace jrb::adapters::camera

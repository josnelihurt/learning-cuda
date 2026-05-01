#include "src/cpp_accelerator/adapters/camera/camera_detector_impl.h"

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

CameraDetectorImpl::CameraDetectorImpl() {
  // Initialize backends in priority order.
  // Backends are conditionally compiled based on Bazel flags.
#ifdef CAMERA_BACKEND_V4L2_ENABLED
  backends_.push_back(std::make_unique<V4L2Backend>());
#endif

#ifdef CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED
  backends_.push_back(std::make_unique<NvidiaArgusBackend>());
#endif

  // Stub is always available as fallback
  backends_.push_back(std::make_unique<StubBackend>());
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

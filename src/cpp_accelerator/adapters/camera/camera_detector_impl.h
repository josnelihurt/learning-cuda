#pragma once

#include <memory>
#include <vector>

#include "src/cpp_accelerator/adapters/camera/backends/camera_backend.h"
#include "proto/_virtual_imports/accelerator_control_proto/accelerator_control.pb.h"

namespace jrb::adapters::camera {

// Orchestrates camera detection by trying each enabled backend in priority order.
// Each backend is tried independently and results are aggregated (so a device
// detected by V4L2 backend and a device detected by Argus backend will both be
// reported, each with its backend name in the display name).
class CameraDetectorImpl {
 public:
  explicit CameraDetectorImpl();
  ~CameraDetectorImpl();

  // Detects cameras by trying each enabled backend.
  // Returns all cameras detected by all available backends.
  std::vector<cuda_learning::RemoteCameraInfo> DetectCameras(
      const std::vector<int>& sensor_ids);

 private:
  std::vector<std::unique_ptr<CameraBackend>> backends_;
};

}  // namespace jrb::adapters::camera

#include "src/cpp_accelerator/adapters/camera/camera_detector.h"
#include "src/cpp_accelerator/adapters/camera/camera_detector_impl.h"

namespace jrb::adapters::camera {

// Static orchestrator instance
static CameraDetectorImpl g_detector;

std::vector<cuda_learning::RemoteCameraInfo> DetectCameras(
    const std::vector<int>& sensor_ids) {
  return g_detector.DetectCameras(sensor_ids);
}

}  // namespace jrb::adapters::camera

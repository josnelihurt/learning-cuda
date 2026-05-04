// V4L2 backend registration for CameraDetectorImpl.
// Compiled only when --config=v4l2-camera is set (select() in BUILD).

#include "src/cpp_accelerator/adapters/camera/camera_detector_impl.h"

#include "src/cpp_accelerator/adapters/camera/backends/v4l2_backend.h"

namespace jrb::adapters::camera {

void CameraDetectorImpl::RegisterV4L2Backend() {
  backends_.push_back(std::make_unique<V4L2Backend>());
}

}  // namespace jrb::adapters::camera

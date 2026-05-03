// V4L2 backend registration stub for non-V4L2 builds (select() in BUILD).

#include "src/cpp_accelerator/adapters/camera/camera_detector_impl.h"

namespace jrb::adapters::camera {

void CameraDetectorImpl::RegisterV4L2Backend() {}

}  // namespace jrb::adapters::camera

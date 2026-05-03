// V4L2 backend registration stub for non-V4L2 builds (select() in BUILD).

#include "src/cpp_accelerator/adapters/camera/gst_camera_source_impl.h"

namespace jrb::adapters::camera {

void GstCameraSourceImpl::RegisterV4L2Backend() {}

}  // namespace jrb::adapters::camera

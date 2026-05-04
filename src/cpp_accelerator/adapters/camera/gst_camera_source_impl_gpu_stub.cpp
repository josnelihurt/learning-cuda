// GpuFrameProcessor stub for non-Argus builds.
// Compiled on x86 / V4L2 / stub paths (select() in BUILD).

#include "src/cpp_accelerator/adapters/camera/gst_camera_source_impl.h"

namespace jrb::adapters::camera {

GpuFrameProcessor* GstCameraSourceImpl::GetGpuFrameProcessor() {
  return nullptr;
}

}  // namespace jrb::adapters::camera

// Argus backend registration for CameraDetectorImpl.
// Compiled only when --config=nvidia-argus-camera is set (select() in BUILD).

#include "src/cpp_accelerator/adapters/camera/camera_detector_impl.h"

#include "src/cpp_accelerator/adapters/camera/backends/nvidia_argus_backend.h"

namespace jrb::adapters::camera {

void CameraDetectorImpl::RegisterArgusBackend() {
  backends_.push_back(std::make_unique<NvidiaArgusBackend>());
}

}  // namespace jrb::adapters::camera

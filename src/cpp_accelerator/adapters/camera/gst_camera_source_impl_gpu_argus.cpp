// GpuFrameProcessor wiring for the NvidiaArgusBackend path.
// Compiled only when --config=nvidia-argus-camera is set (select() in BUILD).

#include "src/cpp_accelerator/adapters/camera/gst_camera_source_impl.h"

#include "src/cpp_accelerator/adapters/camera/backends/nvidia_argus_backend.h"

namespace jrb::adapters::camera {

GpuFrameProcessor* GstCameraSourceImpl::GetGpuFrameProcessor() {
  if (!active_backend_) return nullptr;
  auto* argus = dynamic_cast<NvidiaArgusBackend*>(active_backend_.get());
  if (!argus) return nullptr;
  return argus->GetGpuFrameProcessor();
}

}  // namespace jrb::adapters::camera

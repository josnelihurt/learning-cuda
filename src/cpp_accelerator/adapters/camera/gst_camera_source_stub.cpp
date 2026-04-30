#include "src/cpp_accelerator/adapters/camera/gst_camera_source.h"

#include <string>

#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

// Stub implementation for platforms where GStreamer camera support is not
// enabled at build time. Camera detection still works via V4L2 ioctl, but
// video streaming requires building with --config=v4l2-camera (x86) or
// targeting aarch64 (Jetson).

struct GstCameraSource::Impl {
  bool running = false;
};

GstCameraSource::GstCameraSource() : impl_(std::make_unique<Impl>()) {}
GstCameraSource::~GstCameraSource() = default;

void GstCameraSource::SetFrameCallback(FrameCallback /*cb*/) {}

bool GstCameraSource::IsRunning() const { return impl_->running; }

bool GstCameraSource::Start(int sensor_id, int /*width*/, int /*height*/,
                             int /*fps*/, std::string* error_message) {
  spdlog::warn("[GstCameraSource] Camera streaming not available on this build. "
               "Rebuild with --config=v4l2-camera to enable USB camera streaming. "
               "(sensor_id={})", sensor_id);
  if (error_message) {
    *error_message = "Camera streaming not compiled in. Use --config=v4l2-camera.";
  }
  return false;
}

void GstCameraSource::Stop() {}

}  // namespace jrb::adapters::camera

#include "src/cpp_accelerator/adapters/camera/gst_camera_source.h"
#include "src/cpp_accelerator/adapters/camera/gst_camera_source_impl.h"

namespace jrb::adapters::camera {

GstCameraSource::GstCameraSource()
    : impl_(std::make_unique<GstCameraSourceImpl>().release(),
            [](void* p) { delete static_cast<GstCameraSourceImpl*>(p); }) {}

GstCameraSource::~GstCameraSource() {
  Stop();
}

void GstCameraSource::SetFrameCallback(FrameCallback cb) {
  auto* impl = static_cast<GstCameraSourceImpl*>(impl_.get());
  impl->SetFrameCallback(std::move(cb));
}

bool GstCameraSource::Start(int sensor_id, int width, int height, int fps,
                             std::string* error_message) {
  auto* impl = static_cast<GstCameraSourceImpl*>(impl_.get());
  return impl->Start(sensor_id, width, height, fps, error_message);
}

void GstCameraSource::Stop() {
  auto* impl = static_cast<GstCameraSourceImpl*>(impl_.get());
  impl->Stop();
}

bool GstCameraSource::IsRunning() const {
  auto* impl = static_cast<const GstCameraSourceImpl*>(impl_.get());
  return impl->IsRunning();
}

}  // namespace jrb::adapters::camera

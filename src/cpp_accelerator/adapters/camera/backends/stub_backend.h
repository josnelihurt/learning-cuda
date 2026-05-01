#pragma once

#include <memory>
#include <string>
#include <vector>

#include "src/cpp_accelerator/adapters/camera/backends/camera_backend.h"

namespace jrb::adapters::camera {

// StubBackend is a fallback implementation that does not require GStreamer.
// It always reports no cameras and fails to start streaming.
// Used when no camera backends are enabled at build time.
class StubBackend : public CameraBackend {
 public:
  StubBackend();
  ~StubBackend() override;

  bool IsAvailable() const override;
  std::vector<cuda_learning::RemoteCameraInfo> DetectCameras(
      const std::vector<int>& sensor_ids) override;
  void SetFrameCallback(FrameCallback cb) override;
  bool Start(int sensor_id, int width, int height, int fps,
             std::string* error_message) override;
  void Stop() override;
  bool IsRunning() const override;
  std::string GetBackendName() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jrb::adapters::camera

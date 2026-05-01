#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/cpp_accelerator/adapters/camera/backends/camera_backend.h"

namespace jrb::adapters::camera {

// V4L2Backend implements camera detection and streaming via V4L2 + GStreamer.
// Works on any Linux platform with V4L2 devices (/dev/video*).
class V4L2Backend : public CameraBackend {
 public:
  V4L2Backend();
  ~V4L2Backend() override;

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

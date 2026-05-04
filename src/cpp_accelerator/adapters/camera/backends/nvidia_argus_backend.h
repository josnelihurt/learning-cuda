#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/cpp_accelerator/adapters/camera/backends/camera_backend.h"

namespace jrb::adapters::camera {

class GpuFrameProcessor;

// NvidiaArgusBackend implements camera detection and streaming via NVIDIA Argus.
// Only available on NVIDIA Jetson platforms with GStreamer nvarguscamerasrc.
class NvidiaArgusBackend : public CameraBackend {
 public:
  NvidiaArgusBackend();
  ~NvidiaArgusBackend() override;

  bool IsAvailable() const override;
  std::vector<cuda_learning::RemoteCameraInfo> DetectCameras(
      const std::vector<int>& sensor_ids) override;
  void SetFrameCallback(FrameCallback cb) override;
  bool Start(int sensor_id, int width, int height, int fps,
             std::string* error_message) override;
  void Stop() override;
  bool IsRunning() const override;
  std::string GetBackendName() const override;
  rtc::binary GrabStillFrame(int* out_width, int* out_height) override;

  // Returns the GpuFrameProcessor created during Start().  Null before Start()
  // or after Stop().  Callers (e.g. BirdWatcher) can register an RgbCallback
  // on it to receive RGBA frames for YOLO inference without H.264 decode.
  GpuFrameProcessor* GetGpuFrameProcessor();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jrb::adapters::camera


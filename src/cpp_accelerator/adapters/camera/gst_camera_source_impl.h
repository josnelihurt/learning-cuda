#pragma once

#include <memory>
#include <string>

#include "src/cpp_accelerator/adapters/camera/backends/camera_backend.h"
#include <rtc/rtc.hpp>

namespace jrb::adapters::camera {

class GpuFrameProcessor;

class GpuFrameProcessor;

// Orchestrates camera streaming by trying each enabled backend in priority order.
// The first backend that successfully starts streaming is used; if it fails,
// the next backend is tried.
class GstCameraSourceImpl {
 public:
  using FrameCallback = std::function<void(rtc::binary data, rtc::FrameInfo info)>;

  GstCameraSourceImpl();
  ~GstCameraSourceImpl();

  void SetFrameCallback(FrameCallback cb);
  bool Start(int sensor_id, int width, int height, int fps,
             std::string* error_message);
  void Stop();
  bool IsRunning() const;
  rtc::binary GrabStillFrame(int* out_width, int* out_height);

  // Returns the active backend's GpuFrameProcessor if the backend is
  // NvidiaArgusBackend and streaming has started; nullptr otherwise.
  GpuFrameProcessor* GetGpuFrameProcessor();

 private:
  std::vector<std::unique_ptr<CameraBackend>> backends_;
  std::unique_ptr<CameraBackend> active_backend_;
};

}  // namespace jrb::adapters::camera

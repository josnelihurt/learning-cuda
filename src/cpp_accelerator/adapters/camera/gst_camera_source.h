#pragma once

#include <functional>
#include <memory>
#include <string>

#include <rtc/rtc.hpp>

namespace jrb::adapters::camera {

// GstCameraSource captures from a Jetson camera sensor via GStreamer
// nvarguscamerasrc and delivers encoded H.264 access units via callback.
// On non-Jetson platforms (no nvarguscamerasrc) Start() returns false.
class GstCameraSource {
 public:
  using FrameCallback = std::function<void(rtc::binary data, rtc::FrameInfo info)>;

  GstCameraSource();
  ~GstCameraSource();

  // Non-copyable, non-movable.
  GstCameraSource(const GstCameraSource&) = delete;
  GstCameraSource& operator=(const GstCameraSource&) = delete;

  // Sets the callback invoked for each encoded access unit.
  void SetFrameCallback(FrameCallback cb);

  // Starts the GStreamer pipeline for the given sensor.
  // Returns false if nvarguscamerasrc is unavailable or pipeline fails.
  bool Start(int sensor_id, int width, int height, int fps, std::string* error_message);

  // Stops and tears down the pipeline. Safe to call multiple times.
  void Stop();

  bool IsRunning() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jrb::adapters::camera

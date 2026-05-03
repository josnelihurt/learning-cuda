#pragma once

#include <functional>
#include <memory>
#include <string>

#include <rtc/rtc.hpp>

namespace jrb::adapters::camera {

// GstCameraSource captures video from cameras via GStreamer.
// Tries multiple backends (V4L2, NVIDIA Argus) in priority order until one succeeds.
// Delivers encoded H.264 access units via callback.
// Returns false if no camera backends are enabled or if all backends fail to start.
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

  // Starts the camera streaming pipeline for the given sensor.
  // Tries each enabled backend in priority order until one succeeds.
  // Returns false if no backends are available or all backends fail to start.
  bool Start(int sensor_id, int width, int height, int fps, std::string* error_message);

  // Stops and tears down the pipeline. Safe to call multiple times.
  void Stop();

  bool IsRunning() const;

  // Pulls one full-resolution NV12 frame from the active backend (blocking ≤500 ms).
  // Returns empty vector if the backend doesn't support still capture.
  rtc::binary GrabStillFrame(int* out_width, int* out_height);

 private:
  std::unique_ptr<void, void(*)(void*)> impl_;
};

}  // namespace jrb::adapters::camera

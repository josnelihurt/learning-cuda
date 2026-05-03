#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "proto/_virtual_imports/accelerator_control_proto/accelerator_control.pb.h"
#include <rtc/rtc.hpp>

namespace jrb::adapters::camera {

// Abstract base class for camera backend implementations.
// Each backend (V4L2, NVIDIA Argus, etc.) provides detection and streaming
// through this interface.
class CameraBackend {
 public:
  using FrameCallback = std::function<void(rtc::binary data, rtc::FrameInfo info)>;

  virtual ~CameraBackend() = default;

  // Returns true if this backend is available on the current platform.
  // Called to determine if the backend should be tried.
  virtual bool IsAvailable() const = 0;

  // Detects cameras available through this backend.
  // Returns a vector of RemoteCameraInfo for each detected camera.
  // Returns empty vector if no cameras are available.
  virtual std::vector<cuda_learning::RemoteCameraInfo> DetectCameras(
      const std::vector<int>& sensor_ids) = 0;

  // Sets the callback invoked for each encoded access unit.
  virtual void SetFrameCallback(FrameCallback cb) = 0;

  // Starts streaming from the given sensor.
  // Returns true on success, false otherwise.
  // error_message is populated on failure.
  virtual bool Start(int sensor_id, int width, int height, int fps,
                     std::string* error_message) = 0;

  // Stops and tears down the camera pipeline.
  virtual void Stop() = 0;

  // Returns true if currently streaming.
  virtual bool IsRunning() const = 0;

  // Returns the backend name (e.g., "V4L2", "NVIDIA Argus", "Stub").
  // Used for device identification in UI (e.g., "V4L2: /dev/video0").
  virtual std::string GetBackendName() const = 0;

  // Pulls one full-resolution NV12 frame from the camera (blocking, ≤500 ms).
  // Only NvidiaArgusBackend implements this; all others return an empty vector.
  // out_width / out_height are set to the frame dimensions on success.
  virtual rtc::binary GrabStillFrame(int* out_width, int* out_height) { return {}; }
};

}  // namespace jrb::adapters::camera

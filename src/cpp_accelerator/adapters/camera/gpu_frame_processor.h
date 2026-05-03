#pragma once

// GPU-space NV12 frame processing pipeline for Jetson Orin Nano.
// Orchestrates: NVMM map → NV12→RGBA (CUDA) → optional GPU filters →
//               RGBA→NV12 (CUDA) → EncodePipeline (H.264 for WebRTC)
// Optionally downloads RGBA to host for BirdWatcher YOLO inference.
//
// Only compiled when CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED is defined.

#ifdef CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gst/gst.h>
#include <rtc/rtc.hpp>

namespace jrb::adapters::camera {

// Processes one 720p NV12 NVMM frame per call to Process():
//   1. Map NVMM GstBuffer to CUDA device pointers (NvBufSurface).
//   2. Run NV12→RGBA kernel on device.
//   3. Run optional GPU filter kernels (grayscale / blur) on device.
//   4. Run RGBA→NV12 kernel on device.
//   5. Download processed NV12 to host and push to EncodePipeline → H.264 → FrameCallback.
//   6. If RgbCallback is set: download RGBA to host and call RgbCallback.
//
// Thread safety: Process() may be called from any single thread (the GstAppSink
// callback thread).  The FrameCallback and RgbCallback will be invoked from that
// same thread.
class GpuFrameProcessor {
 public:
  // H.264 output callback (passed to EncodePipeline and forwarded to CameraHub / WebRTC).
  using FrameCallback =
      std::function<void(const rtc::binary& data, const rtc::FrameInfo& info)>;

  // Optional: called with host RGBA (4-channel, width × height × 4 bytes) for
  // YOLO inference or other CPU-side processing.
  using RgbCallback =
      std::function<void(const std::vector<uint8_t>& rgba, int width, int height)>;

  GpuFrameProcessor();
  ~GpuFrameProcessor();

  GpuFrameProcessor(const GpuFrameProcessor&) = delete;
  GpuFrameProcessor& operator=(const GpuFrameProcessor&) = delete;

  // Start the internal EncodePipeline.  width/height are the NV12 frame dimensions;
  // fps is the sensor frame rate (used for caps negotiation).
  bool Start(int width, int height, int fps, FrameCallback h264_cb,
             std::string* error_message);
  void Stop();
  bool IsRunning() const;

  // Register an optional callback to receive host RGBA frames (e.g. for YOLO).
  // Pass nullptr to disable.  May be called before or after Start().
  void SetRgbCallback(RgbCallback cb);

  // Process one NVMM NV12 GstBuffer.  Called from NvidiaArgusBackend::OnNewSample.
  void Process(GstBuffer* nvmm_buf, uint32_t rtp_ts);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jrb::adapters::camera

#endif  // CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED

#pragma once

// GStreamer re-encode pipeline: NV12 host frames → H.264 → FrameCallback.
// Used by GpuFrameProcessor to restore the H.264 stream for WebRTC after
// GPU-space processing on the NV12 frames.
//
// Only compiled when CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED is defined.

#ifdef CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include <gst/gst.h>
#include <rtc/rtc.hpp>

namespace jrb::adapters::camera {

// Wraps a GStreamer pipeline:
//   appsrc → nvvidconv → x264enc → h264parse → appsink
//
// Accepts raw NV12 frames via PushFrame() and delivers H.264 AUs to the
// registered FrameCallback.  The pipeline is dimensioned once at Start() and
// cannot be resized afterwards.
class EncodePipeline {
 public:
  using FrameCallback =
      std::function<void(const rtc::binary& data, const rtc::FrameInfo& info)>;

  EncodePipeline();
  ~EncodePipeline();

  EncodePipeline(const EncodePipeline&) = delete;
  EncodePipeline& operator=(const EncodePipeline&) = delete;

  // Start the pipeline.  width/height match the NV12 frames that will be
  // pushed via PushFrame().  fps is the frame rate for the caps negotiation.
  // The FrameCallback is called for each encoded H.264 AU.
  bool Start(int width, int height, int fps, FrameCallback cb, std::string* error_message);
  void Stop();
  bool IsRunning() const;

  // Push one NV12 frame to the encoder.  nv12_host must contain
  // (width * height * 3 / 2) bytes in packed NV12 layout (stride == width).
  // rtp_ts is propagated as the PTS on the encoded buffer.
  // Returns false if the pipeline is not running or the push fails.
  bool PushFrame(const uint8_t* nv12_host, int width, int height, uint32_t rtp_ts);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jrb::adapters::camera

#endif  // CAMERA_BACKEND_NVIDIA_ARGUS_ENABLED

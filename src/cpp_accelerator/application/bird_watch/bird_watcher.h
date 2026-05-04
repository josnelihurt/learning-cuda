#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <rtc/rtc.hpp>

#include "src/cpp_accelerator/adapters/camera/camera_hub.h"
#include "src/cpp_accelerator/adapters/compute/cuda/memory/cuda_memory_pool.h"

struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace jrb::application::engine {
class ProcessorEngine;
}

namespace jrb::domain::interfaces {
class IImageSink;
}

namespace jrb::adapters::camera {
class CameraHub;
class GpuFrameProcessor;
}

namespace jrb::application::bird_watch {

struct BirdWatcherConfig {
  bool enabled = false;
  float confidence_threshold = 0.4f;
  int idle_interval_s = 3;
  int alert_frames = 5;
  int max_per_minute = 5;
  int min_save_interval_s = 5;
  int camera_sensor_id = 0;
  int capture_width = 1280;
  int capture_height = 720;
  int capture_fps = 30;
  std::string captures_dir;
};

class BirdWatcher {
 public:
  BirdWatcher(BirdWatcherConfig config, std::shared_ptr<jrb::adapters::camera::CameraHub> camera_hub,
              jrb::application::engine::ProcessorEngine* engine,
              jrb::domain::interfaces::IImageSink* image_sink);
  ~BirdWatcher();

  BirdWatcher(const BirdWatcher&) = delete;
  BirdWatcher& operator=(const BirdWatcher&) = delete;

  void Start();
  void Stop();
  bool IsSubscribed() const { return subscription_.IsActive(); }

 private:
  // H.264 callback: only used on x86 (no GPU processor).
  void OnH264Frame(const rtc::binary& data, const rtc::FrameInfo& info);
  // GPU RGBA callback: used on Jetson / NvidiaArgusBackend path.
  void OnRgbaFrame(const std::vector<uint8_t>& rgba, int width, int height);

  void ConnectGpuPath();
  void DisconnectGpuPath();

  void WorkerLoop();
  void ProcessQueuedFrame(rtc::binary data, rtc::FrameInfo info);
  void ProcessRgbaFrame(std::vector<uint8_t> rgba, int width, int height);

  // Always pushes the AU into the decoder. When `want_rgb` is true, also runs
  // sws_scale on the latest decoded frame and fills `rgb` / dims. Returns true
  // if at least one frame was decoded (regardless of whether RGB was requested).
  bool FeedDecoderAndExtractRgb(const rtc::binary& access_unit, bool want_rgb,
                                std::vector<uint8_t>* rgb, int* width, int* height);
  bool RgbFromDecodedFrame(std::vector<uint8_t>* rgb, int* width, int* height);
  bool ShouldRunInferenceNow();
  bool DetectBird(const std::vector<uint8_t>& rgb, int width, int height);
  bool RateGateAllowsSave();
  void MaybeSave(const std::vector<uint8_t>& rgb, int width, int height);
  void SaveCapture(const std::vector<uint8_t>& rgb, int width, int height);
  void InitDecoder();
  void DestroyDecoder();

  enum class State { Idle, Alert };

  BirdWatcherConfig config_;
  std::shared_ptr<jrb::adapters::camera::CameraHub> camera_hub_;
  jrb::application::engine::ProcessorEngine* engine_;
  jrb::domain::interfaces::IImageSink* image_sink_;
  std::unique_ptr<jrb::adapters::compute::cuda::CudaMemoryPool> cuda_memory_pool_;

  jrb::adapters::camera::CameraHub::Subscription subscription_;

  // Non-null when the active backend is NvidiaArgusBackend (Jetson only).
  // In that case the GPU path is used: RGB arrives via RgbCallback, not H.264 decode.
  // This is a non-owning raw pointer; the processor is owned by NvidiaArgusBackend.
  jrb::adapters::camera::GpuFrameProcessor* gpu_processor_ = nullptr;

  std::thread worker_thread_;

  // H.264 frame queue — used only when gpu_processor_ == nullptr.
  std::queue<std::pair<rtc::binary, rtc::FrameInfo>> frame_queue_;

  // RGBA frame queue — used only when gpu_processor_ != nullptr.
  struct RgbaItem {
    std::vector<uint8_t> rgba;
    int width;
    int height;
  };
  std::queue<RgbaItem> rgba_queue_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::atomic<bool> running_{false};
  // Sized to absorb ~1s of 30 fps frames during worker startup so the very
  // first IDR (carrying SPS/PPS) is not dropped before the decoder sees it.
  static constexpr size_t kMaxQueueSize = 60;

  State state_ = State::Idle;
  int consecutive_bird_frames_ = 0;
  int consecutive_no_bird_frames_ = 0;
  std::chrono::steady_clock::time_point last_idle_check_{};

  std::deque<std::chrono::steady_clock::time_point> capture_times_;
  std::chrono::steady_clock::time_point last_capture_time_{};
  bool had_capture_ = false;

  AVCodecContext* decoder_context_ = nullptr;
  AVFrame* decoded_frame_ = nullptr;
  AVFrame* rgb_input_frame_ = nullptr;
  AVPacket* decode_packet_ = nullptr;
  SwsContext* decode_to_rgb_context_ = nullptr;
  int frame_width_ = 0;
  int frame_height_ = 0;
  int input_pixel_format_ = -1;
  bool first_decode_logged_ = false;
};

}  // namespace jrb::application::bird_watch

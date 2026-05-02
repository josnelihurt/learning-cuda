#include "src/cpp_accelerator/application/bird_watch/bird_watcher.h"

#include <ctime>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include <spdlog/spdlog.h>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#include "src/cpp_accelerator/application/engine/processor_engine.h"
#include "src/cpp_accelerator/domain/interfaces/image_sink.h"

namespace jrb::application::bird_watch {

namespace {

using cuda_learning::ACCELERATOR_TYPE_CUDA;
using cuda_learning::FILTER_TYPE_MODEL_INFERENCE;
using cuda_learning::ProcessImageRequest;
using cuda_learning::ProcessImageResponse;

std::string AvErrorToString(int error_code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(error_code, buffer, sizeof(buffer));
  return std::string(buffer);
}

}  // namespace

BirdWatcher::BirdWatcher(BirdWatcherConfig config,
                         std::shared_ptr<jrb::adapters::camera::CameraHub> camera_hub,
                         jrb::application::engine::ProcessorEngine* engine,
                         jrb::domain::interfaces::IImageSink* image_sink)
    : config_(std::move(config)),
      camera_hub_(std::move(camera_hub)),
      engine_(engine),
      image_sink_(image_sink),
      cuda_memory_pool_(std::make_unique<jrb::adapters::compute::cuda::CudaMemoryPool>()) {}

BirdWatcher::~BirdWatcher() { Stop(); }

void BirdWatcher::Start() {
  if (!config_.enabled) {
    return;
  }
  if (running_.exchange(true)) {
    return;
  }
  last_idle_check_ =
      std::chrono::steady_clock::now() - std::chrono::seconds(config_.idle_interval_s);
  state_ = State::Idle;
  consecutive_bird_frames_ = 0;
  consecutive_no_bird_frames_ = 0;

  InitDecoder();

  std::string sub_err;
  subscription_ = camera_hub_->Subscribe(
      config_.camera_sensor_id, config_.capture_width, config_.capture_height, config_.capture_fps,
      [this](const rtc::binary& data, const rtc::FrameInfo& info) { OnH264Frame(data, info); },
      &sub_err);
  if (!subscription_.IsActive()) {
    spdlog::error("[BirdWatcher] CameraHub subscribe failed: {}", sub_err);
    running_ = false;
    DestroyDecoder();
    return;
  }

  worker_thread_ = std::thread([this] { WorkerLoop(); });
}

void BirdWatcher::Stop() {
  if (!running_.exchange(false)) {
    subscription_.Reset();
    DestroyDecoder();
    return;
  }
  queue_cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  subscription_.Reset();
  DestroyDecoder();
}

void BirdWatcher::OnH264Frame(const rtc::binary& data, const rtc::FrameInfo& info) {
  if (!running_.load()) {
    return;
  }
  std::lock_guard<std::mutex> lk(queue_mutex_);
  if (frame_queue_.size() >= kMaxQueueSize) {
    frame_queue_.pop();
  }
  frame_queue_.emplace(rtc::binary(data), info);
  queue_cv_.notify_one();
}

void BirdWatcher::WorkerLoop() {
  while (running_.load()) {
    std::optional<std::pair<rtc::binary, rtc::FrameInfo>> item;
    {
      std::unique_lock<std::mutex> lk(queue_mutex_);
      queue_cv_.wait(lk, [&] { return !running_.load() || !frame_queue_.empty(); });
      if (!running_.load()) {
        break;
      }
      if (frame_queue_.empty()) {
        continue;
      }
      item.emplace(std::move(frame_queue_.front()));
      frame_queue_.pop();
    }
    if (item) {
      ProcessQueuedFrame(std::move(item->first), std::move(item->second));
    }
  }
}

bool BirdWatcher::ShouldRunInferenceNow() {
  if (state_ == State::Alert) {
    return true;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now - last_idle_check_ >= std::chrono::seconds(config_.idle_interval_s)) {
    last_idle_check_ = now;
    return true;
  }
  return false;
}

void BirdWatcher::ProcessQueuedFrame(rtc::binary data, rtc::FrameInfo info) {
  (void)info;
  // The H.264 stream from GstCameraSource only emits SPS/PPS at the very first
  // IDR (x264enc default). Skipping any AU permanently desyncs the decoder, so
  // every access unit is fed; YOLO inference is the only thing rate-limited.
  const bool want_rgb = ShouldRunInferenceNow();
  std::vector<uint8_t> rgb;
  int w = 0;
  int h = 0;
  if (!FeedDecoderAndExtractRgb(data, want_rgb, &rgb, &w, &h)) {
    spdlog::debug("[BirdWatcher] decoder warming up (au {} bytes)", data.size());
    return;
  }
  if (!want_rgb) {
    return;
  }

  const bool bird = DetectBird(rgb, w, h);
  if (state_ == State::Idle) {
    spdlog::info("[BirdWatcher] idle check {}x{} bird={}", w, h, bird);
    if (bird) {
      spdlog::info("[BirdWatcher] IDLE -> ALERT (bird detected)");
      state_ = State::Alert;
      consecutive_bird_frames_ = 0;
      consecutive_no_bird_frames_ = 0;
    }
    return;
  }

  if (bird) {
    consecutive_bird_frames_++;
    consecutive_no_bird_frames_ = 0;
    if (consecutive_bird_frames_ >= config_.alert_frames) {
      MaybeSave(rgb, w, h);
      consecutive_bird_frames_ = 0;
    }
  } else {
    consecutive_bird_frames_ = 0;
    consecutive_no_bird_frames_++;
    if (consecutive_no_bird_frames_ >= config_.alert_frames) {
      spdlog::info("[BirdWatcher] ALERT -> IDLE (no bird)");
      state_ = State::Idle;
      consecutive_no_bird_frames_ = 0;
      last_idle_check_ = std::chrono::steady_clock::now();
    }
  }
  spdlog::debug("[BirdWatcher] alert frame {}x{} bird={} bird_streak={} no_bird_streak={}", w, h,
                bird, consecutive_bird_frames_, consecutive_no_bird_frames_);
}

void BirdWatcher::InitDecoder() {
  if (decoder_context_ != nullptr) {
    return;
  }
  if (decoded_frame_ == nullptr) {
    decoded_frame_ = av_frame_alloc();
  }
  if (rgb_input_frame_ == nullptr) {
    rgb_input_frame_ = av_frame_alloc();
  }
  if (decode_packet_ == nullptr) {
    decode_packet_ = av_packet_alloc();
  }
  if (!decoded_frame_ || !rgb_input_frame_ || !decode_packet_) {
    spdlog::error("[BirdWatcher] FFmpeg alloc failed");
    return;
  }

  const AVCodec* decoder = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (decoder == nullptr) {
    spdlog::error("[BirdWatcher] H264 decoder not available");
    return;
  }
  decoder_context_ = avcodec_alloc_context3(decoder);
  if (decoder_context_ == nullptr) {
    spdlog::error("[BirdWatcher] avcodec_alloc_context3 failed");
    return;
  }
  const int open_result = avcodec_open2(decoder_context_, decoder, nullptr);
  if (open_result < 0) {
    spdlog::error("[BirdWatcher] avcodec_open2: {}", AvErrorToString(open_result));
    avcodec_free_context(&decoder_context_);
    decoder_context_ = nullptr;
    return;
  }
  first_decode_logged_ = false;
}

void BirdWatcher::DestroyDecoder() {
  sws_freeContext(decode_to_rgb_context_);
  decode_to_rgb_context_ = nullptr;
  av_frame_free(&rgb_input_frame_);
  av_frame_free(&decoded_frame_);
  av_packet_free(&decode_packet_);
  avcodec_free_context(&decoder_context_);
  frame_width_ = 0;
  frame_height_ = 0;
  input_pixel_format_ = -1;
}

bool BirdWatcher::FeedDecoderAndExtractRgb(const rtc::binary& access_unit, bool want_rgb,
                                           std::vector<uint8_t>* rgb, int* width, int* height) {
  if (decoder_context_ == nullptr || decode_packet_ == nullptr) {
    return false;
  }
  if (access_unit.empty()) {
    return false;
  }

  // GstCameraSource pins Annex-B byte-stream + AU alignment, so each callback
  // delivers exactly one access unit ready to be fed verbatim — same pattern
  // as LiveVideoProcessor::ProcessAccessUnit.
  av_packet_unref(decode_packet_);
  decode_packet_->data =
      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(access_unit.data()));
  decode_packet_->size = static_cast<int>(access_unit.size());

  const int send_result = avcodec_send_packet(decoder_context_, decode_packet_);
  if (send_result < 0) {
    spdlog::debug("[BirdWatcher] avcodec_send_packet: {}", AvErrorToString(send_result));
    return false;
  }

  bool got_picture = false;
  while (true) {
    const int recv = avcodec_receive_frame(decoder_context_, decoded_frame_);
    if (recv == AVERROR(EAGAIN) || recv == AVERROR_EOF) {
      break;
    }
    if (recv < 0) {
      spdlog::debug("[BirdWatcher] avcodec_receive_frame: {}", AvErrorToString(recv));
      av_frame_unref(decoded_frame_);
      break;
    }
    if (!first_decode_logged_) {
      spdlog::info("[BirdWatcher] first H264 frame decoded {}x{}", decoded_frame_->width,
                   decoded_frame_->height);
      first_decode_logged_ = true;
    }
    if (want_rgb && rgb != nullptr && width != nullptr && height != nullptr) {
      if (RgbFromDecodedFrame(rgb, width, height)) {
        got_picture = true;
      }
    } else {
      got_picture = true;
    }
    av_frame_unref(decoded_frame_);
  }
  return got_picture;
}

bool BirdWatcher::RgbFromDecodedFrame(std::vector<uint8_t>* rgb, int* width, int* height) {
  const int w = decoded_frame_->width;
  const int h = decoded_frame_->height;
  const int input_format = decoded_frame_->format;
  if (w <= 0 || h <= 0) {
    return false;
  }

  const bool needs_recreate =
      w != frame_width_ || h != frame_height_ || input_format != input_pixel_format_;
  if (needs_recreate) {
    sws_freeContext(decode_to_rgb_context_);
    decode_to_rgb_context_ = nullptr;
    av_frame_unref(rgb_input_frame_);
    rgb_input_frame_->format = AV_PIX_FMT_RGB24;
    rgb_input_frame_->width = w;
    rgb_input_frame_->height = h;
    if (av_frame_get_buffer(rgb_input_frame_, 32) < 0) {
      spdlog::error("[BirdWatcher] av_frame_get_buffer RGB failed");
      return false;
    }
    decode_to_rgb_context_ = sws_getCachedContext(
        nullptr, w, h, static_cast<AVPixelFormat>(input_format), w, h, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (decode_to_rgb_context_ == nullptr) {
      spdlog::error("[BirdWatcher] sws_getCachedContext failed");
      return false;
    }
    frame_width_ = w;
    frame_height_ = h;
    input_pixel_format_ = input_format;
  }

  if (av_frame_make_writable(rgb_input_frame_) < 0) {
    return false;
  }
  if (sws_scale(decode_to_rgb_context_, decoded_frame_->data, decoded_frame_->linesize, 0, h,
                 rgb_input_frame_->data, rgb_input_frame_->linesize) <= 0) {
    return false;
  }

  rgb->assign(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u, 0);
  for (int row = 0; row < h; ++row) {
    const auto* src = rgb_input_frame_->data[0] + row * rgb_input_frame_->linesize[0];
    auto* dst = rgb->data() + static_cast<size_t>(row) * static_cast<size_t>(w) * 3u;
    std::memcpy(dst, src, static_cast<size_t>(w) * 3u);
  }
  *width = w;
  *height = h;
  return true;
}

bool BirdWatcher::DetectBird(const std::vector<uint8_t>& rgb, int width, int height) {
  if (engine_ == nullptr) {
    return false;
  }
  ProcessImageRequest req;
  req.set_accelerator(ACCELERATOR_TYPE_CUDA);
  req.add_filters(FILTER_TYPE_MODEL_INFERENCE);
  req.mutable_model_params()->set_model_id("yolov10n");
  req.mutable_model_params()->set_confidence_threshold(config_.confidence_threshold);
  req.set_image_data(reinterpret_cast<const char*>(rgb.data()), rgb.size());
  req.set_width(width);
  req.set_height(height);
  req.set_channels(3);
  req.set_api_version("1.1");

  ProcessImageResponse resp;
  if (!engine_->ProcessImage(req, &resp, cuda_memory_pool_.get()) || resp.code() != 0) {
    spdlog::warn("[BirdWatcher] ProcessImage failed: {}", resp.message());
    return false;
  }
  bool hit = false;
  for (const auto& d : resp.detections()) {
    if (d.class_name() == "bird" && d.confidence() >= config_.confidence_threshold) {
      hit = true;
      break;
    }
  }
  spdlog::debug("[BirdWatcher] YOLO {}x{} threshold={:.2f} raw_dets={} bird_hit={}", width, height,
                config_.confidence_threshold, resp.detections_size(), hit);
  return hit;
}

bool BirdWatcher::RateGateAllowsSave() {
  const auto now = std::chrono::steady_clock::now();
  while (!capture_times_.empty() &&
         now - capture_times_.front() > std::chrono::seconds(60)) {
    capture_times_.pop_front();
  }
  if (capture_times_.size() >= static_cast<size_t>(config_.max_per_minute)) {
    return false;
  }
  if (had_capture_ &&
      now - last_capture_time_ < std::chrono::seconds(config_.min_save_interval_s)) {
    return false;
  }
  return true;
}

void BirdWatcher::MaybeSave(const std::vector<uint8_t>& rgb, int width, int height) {
  if (!RateGateAllowsSave()) {
    return;
  }
  SaveCapture(rgb, width, height);
}

void BirdWatcher::SaveCapture(const std::vector<uint8_t>& rgb, int width, int height) {
  if (image_sink_ == nullptr || config_.captures_dir.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(config_.captures_dir, ec);
  if (ec) {
    spdlog::error("[BirdWatcher] create_directories {}: {}", config_.captures_dir, ec.message());
    return;
  }

  const auto sys_now = std::chrono::system_clock::now();
  const std::time_t tt = std::chrono::system_clock::to_time_t(sys_now);
  std::tm tm_buf{};
  gmtime_r(&tt, &tm_buf);
  std::ostringstream name;
  name << std::put_time(&tm_buf, "%Y-%m-%dT%H-%M-%SZ");
  const std::string path = config_.captures_dir + "/" + name.str() + ".bmp";

  if (!image_sink_->writeBmp(path.c_str(), rgb.data(), width, height, 3)) {
    spdlog::error("[BirdWatcher] writeBmp failed: {}", path);
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  had_capture_ = true;
  last_capture_time_ = now;
  capture_times_.push_back(now);
  spdlog::info("[BirdWatcher] Saved capture: {}", path);
}

}  // namespace jrb::application::bird_watch

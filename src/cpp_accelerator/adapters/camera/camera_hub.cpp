#include "src/cpp_accelerator/adapters/camera/camera_hub.h"

#include <utility>

#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

CameraHub::Subscription::Subscription(std::shared_ptr<CameraHub> hub, int sensor_id,
                                      uint64_t token)
    : hub_(std::move(hub)), sensor_id_(sensor_id), token_(token) {}

CameraHub::Subscription::~Subscription() {
  Reset();
}

CameraHub::Subscription::Subscription(Subscription&& other) noexcept
    : hub_(std::move(other.hub_)), sensor_id_(other.sensor_id_), token_(other.token_) {
  other.sensor_id_ = 0;
  other.token_ = 0;
}

CameraHub::Subscription& CameraHub::Subscription::operator=(Subscription&& other) noexcept {
  if (this != &other) {
    Reset();
    hub_ = std::move(other.hub_);
    sensor_id_ = other.sensor_id_;
    token_ = other.token_;
    other.sensor_id_ = 0;
    other.token_ = 0;
  }
  return *this;
}

void CameraHub::Subscription::Reset() {
  if (hub_) {
    hub_->Unsubscribe(sensor_id_, token_);
    hub_.reset();
    sensor_id_ = 0;
    token_ = 0;
  }
}

std::shared_ptr<CameraHub> CameraHub::Create() {
  auto hub = std::shared_ptr<CameraHub>(new CameraHub());
  hub->self_ = hub;
  return hub;
}

CameraHub::~CameraHub() {
  Shutdown();
}

void CameraHub::Shutdown() {
  std::map<int, std::unique_ptr<Stream>> drained;
  {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    drained.swap(streams_);
  }
  for (auto& [sensor_id, stream] : drained) {
    if (stream && stream->source) {
      stream->source->Stop();
      spdlog::info("[CameraHub] Stopped sensor_id={}", sensor_id);
    }
  }
}

CameraHub::Subscription CameraHub::Subscribe(int sensor_id, int width, int height, int fps,
                                             FrameCallback cb, std::string* error_message) {
  if (!cb) {
    if (error_message) *error_message = "callback is null";
    return Subscription();
  }

  Stream* stream_ptr = nullptr;
  uint64_t token = 0;
  bool need_start = false;

  {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    auto it = streams_.find(sensor_id);
    if (it == streams_.end()) {
      auto stream = std::make_unique<Stream>();
      stream->source = std::make_unique<GstCameraSource>();
      stream->width = width;
      stream->height = height;
      stream->fps = fps;
      stream_ptr = stream.get();
      streams_.emplace(sensor_id, std::move(stream));
      need_start = true;
    } else {
      stream_ptr = it->second.get();
    }

    token = next_token_++;
    {
      std::lock_guard<std::mutex> sub_lock(stream_ptr->subscribers_mutex);
      stream_ptr->subscribers.emplace(token, std::move(cb));
    }
  }

  if (need_start) {
    // Wire fan-out callback before starting so frames are never dropped.
    auto weak_self = self_;
    const int sid = sensor_id;
    stream_ptr->source->SetFrameCallback(
        [weak_self, sid](rtc::binary data, rtc::FrameInfo info) {
          auto hub = weak_self.lock();
          if (!hub) return;
          Stream* s = nullptr;
          {
            std::lock_guard<std::mutex> lock(hub->streams_mutex_);
            auto it = hub->streams_.find(sid);
            if (it == hub->streams_.end()) return;
            s = it->second.get();
          }
          // Snapshot subscribers to avoid holding the lock during callbacks.
          std::vector<FrameCallback> callbacks;
          {
            std::lock_guard<std::mutex> sub_lock(s->subscribers_mutex);
            callbacks.reserve(s->subscribers.size());
            for (const auto& [tok, fn] : s->subscribers) {
              callbacks.push_back(fn);
            }
          }
          for (const auto& fn : callbacks) {
            try {
              fn(data, info);
            } catch (const std::exception& e) {
              spdlog::warn("[CameraHub:{}] Subscriber threw: {}", sid, e.what());
            }
          }
        });

    std::string err;
    if (!stream_ptr->source->Start(sensor_id, width, height, fps, &err)) {
      if (error_message) *error_message = err;
      spdlog::error("[CameraHub] Failed to start sensor_id={}: {}", sensor_id, err);
      // Roll back: drop this stream and the just-added subscriber.
      std::lock_guard<std::mutex> lock(streams_mutex_);
      streams_.erase(sensor_id);
      return Subscription();
    }
    spdlog::info("[CameraHub] Started sensor_id={} {}x{}@{}fps", sensor_id, width, height, fps);
  } else {
    spdlog::info("[CameraHub] New subscriber for sensor_id={} (already running)", sensor_id);
  }

  auto hub_shared = self_.lock();
  if (!hub_shared) {
    // Hub is being destroyed.
    Unsubscribe(sensor_id, token);
    return Subscription();
  }
  return Subscription(std::move(hub_shared), sensor_id, token);
}

void CameraHub::Unsubscribe(int sensor_id, uint64_t token) {
  if (token == 0) return;
  std::lock_guard<std::mutex> lock(streams_mutex_);
  auto it = streams_.find(sensor_id);
  if (it == streams_.end()) return;
  Stream* stream = it->second.get();
  std::lock_guard<std::mutex> sub_lock(stream->subscribers_mutex);
  stream->subscribers.erase(token);
  // Camera stays open even with zero subscribers — kept alive for the process
  // lifetime to avoid V4L2 reopen latency / EBUSY races.
}

}  // namespace jrb::adapters::camera

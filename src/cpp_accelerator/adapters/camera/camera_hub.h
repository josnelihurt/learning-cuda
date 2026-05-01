#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <rtc/rtc.hpp>

#include "src/cpp_accelerator/adapters/camera/gst_camera_source.h"

namespace jrb::adapters::camera {

// CameraHub owns one GstCameraSource per sensor_id for the lifetime of the
// process and fans out encoded H.264 access units to any number of subscribers.
//
// Sessions never own the device — they hold a Subscription RAII handle. The
// underlying V4L2 fd is opened on first subscribe and kept open until the hub
// is destroyed (process shutdown). This eliminates EBUSY on rapid open/close
// cycles and lets multiple sessions share the same camera.
class CameraHub {
 public:
  using FrameCallback = std::function<void(const rtc::binary& data, const rtc::FrameInfo& info)>;

  // RAII handle returned by Subscribe(). Destruction unregisters the callback.
  class Subscription {
   public:
    Subscription() = default;
    Subscription(std::shared_ptr<CameraHub> hub, int sensor_id, uint64_t token);
    ~Subscription();

    Subscription(const Subscription&) = delete;
    Subscription& operator=(const Subscription&) = delete;
    Subscription(Subscription&& other) noexcept;
    Subscription& operator=(Subscription&& other) noexcept;

    void Reset();
    bool IsActive() const { return hub_ != nullptr; }

   private:
    std::shared_ptr<CameraHub> hub_;
    int sensor_id_{0};
    uint64_t token_{0};
  };

  static std::shared_ptr<CameraHub> Create();
  ~CameraHub();

  CameraHub(const CameraHub&) = delete;
  CameraHub& operator=(const CameraHub&) = delete;

  // Subscribe to encoded frames from the given sensor. Lazily opens the device
  // on the first subscription. Returns an inactive Subscription on failure
  // (and populates *error_message if provided).
  Subscription Subscribe(int sensor_id, int width, int height, int fps,
                         FrameCallback cb, std::string* error_message = nullptr);

  // Tears down all camera streams. Called automatically on destruction.
  void Shutdown();

 private:
  CameraHub() = default;

  struct Stream {
    std::unique_ptr<GstCameraSource> source;
    std::mutex subscribers_mutex;
    std::unordered_map<uint64_t, FrameCallback> subscribers;
    int width{0};
    int height{0};
    int fps{0};
  };

  void Unsubscribe(int sensor_id, uint64_t token);

  std::mutex streams_mutex_;
  std::map<int, std::unique_ptr<Stream>> streams_;
  uint64_t next_token_{1};
  std::weak_ptr<CameraHub> self_;
};

}  // namespace jrb::adapters::camera

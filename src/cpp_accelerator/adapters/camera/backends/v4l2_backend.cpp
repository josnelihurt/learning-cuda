#include "src/cpp_accelerator/adapters/camera/backends/v4l2_backend.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/videodev2.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

namespace {

// Returns the card name from V4L2 capability query, or empty string on failure.
std::string QueryV4L2DeviceName(const std::string& device_path) {
  int fd = open(device_path.c_str(), O_RDWR | O_NONBLOCK);
  if (fd < 0) return "";

  v4l2_capability cap{};
  if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
    close(fd);
    return "";
  }

  // Only interested in video capture devices.
  if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    close(fd);
    return "";
  }

  close(fd);
  return reinterpret_cast<const char*>(cap.card);
}

}  // namespace

struct V4L2Backend::Impl {
  FrameCallback cb;
  std::atomic<bool> running{false};
  GstElement* pipeline{nullptr};
  GstElement* appsink{nullptr};
  std::thread bus_thread;

  static GstFlowReturn OnNewSample(GstAppSink* sink, gpointer user_data) {
    auto* impl = static_cast<Impl*>(user_data);
    if (!impl->running.load()) return GST_FLOW_OK;

    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer* buf = gst_sample_get_buffer(sample);
    if (!buf) {
      gst_sample_unref(sample);
      return GST_FLOW_OK;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
      gst_sample_unref(sample);
      return GST_FLOW_OK;
    }

    rtc::binary data(reinterpret_cast<const std::byte*>(map.data),
                     reinterpret_cast<const std::byte*>(map.data) + map.size);

    const GstClockTime pts = GST_BUFFER_PTS(buf);
    const uint32_t rtp_ts = GST_CLOCK_TIME_IS_VALID(pts)
                                ? static_cast<uint32_t>(pts / 1000u)
                                : 0u;
    rtc::FrameInfo info{rtp_ts};

    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);

    if (impl->cb) {
      try {
        impl->cb(std::move(data), info);
      } catch (const std::exception& e) {
        spdlog::warn("[V4L2Backend] Frame callback threw: {}", e.what());
      }
    }
    return GST_FLOW_OK;
  }

  void BusLoop() {
    GstBus* bus = gst_element_get_bus(pipeline);
    while (running.load()) {
      GstMessage* msg = gst_bus_timed_pop_filtered(
          bus,
          100 * GST_MSECOND,
          static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
      if (!msg) continue;
      GstMessageType type = GST_MESSAGE_TYPE(msg);
      if (type == GST_MESSAGE_ERROR) {
        GError* err = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(msg, &err, &debug);
        spdlog::error("[V4L2Backend] Pipeline error: {} ({})",
                      err ? err->message : "unknown",
                      debug ? debug : "");
        if (err) g_error_free(err);
        if (debug) g_free(debug);
        running = false;
      } else if (type == GST_MESSAGE_EOS) {
        spdlog::info("[V4L2Backend] Pipeline EOS");
        running = false;
      }
      gst_message_unref(msg);
    }
    gst_object_unref(bus);

    // Async errors (V4L2 caps negotiation, device-busy, etc.) fire after Start()
    // has already returned ok. If we leave the pipeline in PLAYING/PAUSED state
    // here, v4l2src keeps the device file descriptor open indefinitely and the
    // next StartCameraStream attempt fails with "Device or resource busy". Drop
    // to NULL state from this thread so the FD is released immediately.
    if (pipeline) {
      gst_element_set_state(pipeline, GST_STATE_NULL);
    }
  }
};

V4L2Backend::V4L2Backend() : impl_(std::make_unique<Impl>()) {}

V4L2Backend::~V4L2Backend() {
  Stop();
}

bool V4L2Backend::IsAvailable() const {
  // V4L2 is a Linux kernel API, always available on Linux
  return true;
}

std::vector<cuda_learning::RemoteCameraInfo> V4L2Backend::DetectCameras(
    const std::vector<int>& sensor_ids) {
  std::vector<cuda_learning::RemoteCameraInfo> result;

  for (int sensor_id : sensor_ids) {
    const std::string device_path = "/dev/video" + std::to_string(sensor_id);

    const std::string card_name = QueryV4L2DeviceName(device_path);
    if (card_name.empty()) {
      spdlog::debug("[V4L2Backend] {} not a V4L2 capture device — skipping", device_path);
      continue;
    }

    cuda_learning::RemoteCameraInfo info;
    info.set_sensor_id(sensor_id);
    const std::string display_name =
        "V4L2: " + card_name + " (" + device_path + ")";
    info.set_display_name(display_name);
    info.set_model(card_name);

    auto* mode = info.add_modes();
    mode->set_width(1920);
    mode->set_height(1080);
    mode->set_fps(30.0);

    spdlog::info("[V4L2Backend] Detected camera at {}: {}", device_path, display_name);
    result.push_back(std::move(info));
  }

  return result;
}

void V4L2Backend::SetFrameCallback(FrameCallback cb) {
  impl_->cb = std::move(cb);
}

bool V4L2Backend::Start(int sensor_id, int width, int height, int fps,
                        std::string* error_message) {
  gst_init(nullptr, nullptr);

  const std::string device_path = "/dev/video" + std::to_string(sensor_id);

  // Most USB webcams expose MJPG at high resolutions and only raw YUY2 at low
  // resolutions. We force MJPG negotiation so v4l2src lets the camera pick its
  // best supported size when width/height/fps are 0 (auto). jpegdec then
  // produces raw frames for x264enc.
  std::string mjpg_caps = "image/jpeg";
  if (width > 0 && height > 0) {
    char buf[96];
    snprintf(buf, sizeof(buf), "image/jpeg,width=%d,height=%d", width, height);
    mjpg_caps = buf;
  }
  if (fps > 0) {
    char buf[32];
    snprintf(buf, sizeof(buf), ",framerate=%d/1", fps);
    mjpg_caps += buf;
  }
  const int key_int = fps > 0 ? fps : 30;

  // Output caps pin Annex-B byte-stream + AU alignment so libdatachannel's
  // H264RtpPacketizer (configured with NalUnit::Separator::StartSequence) can
  // packetize directly. This also removes the need for h264parse, which lives
  // in gstreamer1.0-plugins-bad and may not be installed.
  char pipeline_str[768];
  snprintf(pipeline_str, sizeof(pipeline_str),
           "v4l2src device=%s ! %s ! "
           "jpegdec ! videoconvert ! video/x-raw,format=I420 ! "
           "x264enc tune=zerolatency speed-preset=ultrafast "
           "key-int-max=%d bitrate=2000 ! "
           "video/x-h264,profile=constrained-baseline,"
           "stream-format=byte-stream,alignment=au ! "
           "appsink name=sink emit-signals=true max-buffers=2 drop=true",
           device_path.c_str(), mjpg_caps.c_str(), key_int);

  spdlog::info("[V4L2Backend] Launching pipeline: {}", pipeline_str);

  GError* err = nullptr;
  impl_->pipeline = gst_parse_launch(pipeline_str, &err);
  if (!impl_->pipeline) {
    const std::string msg = err ? std::string("Failed to parse pipeline: ") + err->message
                                : "Failed to parse pipeline (unknown error)";
    if (err) g_error_free(err);
    if (error_message) *error_message = msg;
    spdlog::error("[V4L2Backend] {} — is gstreamer1.0-plugins-ugly installed?", msg);
    return false;
  }
  if (err) {
    g_error_free(err);
  }

  impl_->appsink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "sink");
  if (!impl_->appsink) {
    if (error_message) *error_message = "Failed to find appsink element";
    spdlog::error("[V4L2Backend] appsink element not found");
    gst_object_unref(impl_->pipeline);
    impl_->pipeline = nullptr;
    return false;
  }

  GstAppSinkCallbacks callbacks{};
  callbacks.new_sample = &Impl::OnNewSample;
  gst_app_sink_set_callbacks(GST_APP_SINK(impl_->appsink), &callbacks, impl_.get(), nullptr);

  GstStateChangeReturn ret = gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    if (error_message) *error_message = "Failed to set pipeline to PLAYING state";
    spdlog::error("[V4L2Backend] Failed to start pipeline for {}", device_path);
    gst_object_unref(impl_->appsink);
    impl_->appsink = nullptr;
    gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    gst_object_unref(impl_->pipeline);
    impl_->pipeline = nullptr;
    return false;
  }

  impl_->running = true;
  impl_->bus_thread = std::thread([this]() { impl_->BusLoop(); });

  spdlog::info("[V4L2Backend] Started camera {} {}x{}@{}fps",
               device_path, width, height, fps);
  return true;
}

void V4L2Backend::Stop() {
  if (!impl_->running.load() && impl_->pipeline == nullptr) return;

  impl_->running = false;

  if (impl_->bus_thread.joinable()) {
    impl_->bus_thread.join();
  }

  if (impl_->pipeline) {
    gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    if (impl_->appsink) {
      gst_object_unref(impl_->appsink);
      impl_->appsink = nullptr;
    }
    gst_object_unref(impl_->pipeline);
    impl_->pipeline = nullptr;
    spdlog::info("[V4L2Backend] Pipeline stopped and released");
  }
}

bool V4L2Backend::IsRunning() const {
  return impl_->running.load();
}

std::string V4L2Backend::GetBackendName() const {
  return "V4L2";
}

}  // namespace jrb::adapters::camera

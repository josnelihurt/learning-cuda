#include "src/cpp_accelerator/adapters/camera/gst_camera_source.h"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

struct GstCameraSource::Impl {
  GstCameraSource::FrameCallback cb;
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
        spdlog::warn("[GstCameraSource] Frame callback threw: {}", e.what());
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
        spdlog::error("[GstCameraSource] Pipeline error: {} ({})",
                      err ? err->message : "unknown",
                      debug ? debug : "");
        if (err) g_error_free(err);
        if (debug) g_free(debug);
        running = false;
      } else if (type == GST_MESSAGE_EOS) {
        spdlog::info("[GstCameraSource] Pipeline EOS");
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

GstCameraSource::GstCameraSource() : impl_(std::make_unique<Impl>()) {}

GstCameraSource::~GstCameraSource() {
  Stop();
}

void GstCameraSource::SetFrameCallback(FrameCallback cb) {
  impl_->cb = std::move(cb);
}

bool GstCameraSource::IsRunning() const {
  return impl_->running.load();
}

bool GstCameraSource::Start(int sensor_id, int width, int height, int fps,
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

  spdlog::info("[GstCameraSource] Launching V4L2 pipeline: {}", pipeline_str);

  GError* err = nullptr;
  impl_->pipeline = gst_parse_launch(pipeline_str, &err);
  if (!impl_->pipeline) {
    const std::string msg = err ? std::string("Failed to parse pipeline: ") + err->message
                                : "Failed to parse pipeline (unknown error)";
    if (err) g_error_free(err);
    if (error_message) *error_message = msg;
    spdlog::error("[GstCameraSource] {} — is gstreamer1.0-plugins-ugly installed?", msg);
    return false;
  }
  if (err) {
    g_error_free(err);
  }

  impl_->appsink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "sink");
  if (!impl_->appsink) {
    if (error_message) *error_message = "Failed to find appsink element";
    spdlog::error("[GstCameraSource] appsink element not found");
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
    spdlog::error("[GstCameraSource] Failed to start V4L2 pipeline for {}", device_path);
    gst_object_unref(impl_->appsink);
    impl_->appsink = nullptr;
    gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    gst_object_unref(impl_->pipeline);
    impl_->pipeline = nullptr;
    return false;
  }

  impl_->running = true;
  impl_->bus_thread = std::thread([this]() { impl_->BusLoop(); });

  spdlog::info("[GstCameraSource] Started V4L2 camera {} {}x{}@{}fps",
               device_path, width, height, fps);
  return true;
}

void GstCameraSource::Stop() {
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
    spdlog::info("[GstCameraSource] V4L2 pipeline stopped and released");
  }
}

}  // namespace jrb::adapters::camera

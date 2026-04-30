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

    rtc::FrameInfo info{};
    const GstClockTime pts = GST_BUFFER_PTS(buf);
    if (GST_CLOCK_TIME_IS_VALID(pts)) {
      info.timestamp = static_cast<uint32_t>(pts / 1000u);
    }

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

  char pipeline_str[512];
  snprintf(pipeline_str, sizeof(pipeline_str),
           "nvarguscamerasrc sensor-id=%d ! "
           "video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1,format=NV12 ! "
           "nvvidconv ! "
           "nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! "
           "h264parse config-interval=-1 ! "
           "appsink name=sink emit-signals=true max-buffers=2 drop=true",
           sensor_id, width, height, fps);

  spdlog::info("[GstCameraSource] Launching Jetson pipeline: {}", pipeline_str);

  GError* err = nullptr;
  impl_->pipeline = gst_parse_launch(pipeline_str, &err);
  if (!impl_->pipeline) {
    const std::string msg = err ? std::string("Failed to parse pipeline: ") + err->message
                                : "Failed to parse pipeline (unknown error)";
    if (err) g_error_free(err);
    if (error_message) *error_message = msg;
    spdlog::error("[GstCameraSource] {}", msg);
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
    spdlog::error("[GstCameraSource] Failed to start pipeline for sensor-id={}", sensor_id);
    gst_object_unref(impl_->appsink);
    impl_->appsink = nullptr;
    gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    gst_object_unref(impl_->pipeline);
    impl_->pipeline = nullptr;
    return false;
  }

  impl_->running = true;
  impl_->bus_thread = std::thread([this]() { impl_->BusLoop(); });

  spdlog::info("[GstCameraSource] Started Jetson camera sensor-id={} {}x{}@{}fps",
               sensor_id, width, height, fps);
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
    spdlog::info("[GstCameraSource] Pipeline stopped and released");
  }
}

}  // namespace jrb::adapters::camera

#include "src/cpp_accelerator/adapters/camera/encode_pipeline.h"

#include <cstring>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

struct EncodePipeline::Impl {
  FrameCallback cb;
  std::atomic<bool> running{false};
  GstElement* pipeline{nullptr};
  GstElement* appsrc{nullptr};
  GstElement* appsink{nullptr};
  std::thread bus_thread;
  uint64_t frame_count{0};

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
    const uint32_t rtp_ts  = GST_CLOCK_TIME_IS_VALID(pts)
                                 ? static_cast<uint32_t>(pts / 1000u)
                                 : 0u;
    rtc::FrameInfo info{rtp_ts};

    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);

    if (impl->cb) {
      try {
        impl->cb(std::move(data), info);
      } catch (const std::exception& e) {
        spdlog::warn("[EncodePipeline] FrameCallback threw: {}", e.what());
      }
    }
    return GST_FLOW_OK;
  }

  void BusLoop() {
    GstBus* bus = gst_element_get_bus(pipeline);
    while (running.load()) {
      GstMessage* msg = gst_bus_timed_pop_filtered(
          bus, 100 * GST_MSECOND,
          static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
      if (!msg) continue;
      if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* err   = nullptr;
        gchar* debug  = nullptr;
        gst_message_parse_error(msg, &err, &debug);
        spdlog::error("[EncodePipeline] Error: {} ({})", err ? err->message : "?",
                      debug ? debug : "");
        if (err) g_error_free(err);
        if (debug) g_free(debug);
        running = false;
      }
      gst_message_unref(msg);
    }
    gst_object_unref(bus);
  }

  void Cleanup() {
    if (pipeline) {
      gst_element_set_state(pipeline, GST_STATE_NULL);
      if (appsrc) {
        gst_object_unref(appsrc);
        appsrc = nullptr;
      }
      if (appsink) {
        gst_object_unref(appsink);
        appsink = nullptr;
      }
      gst_object_unref(pipeline);
      pipeline = nullptr;
    }
  }
};

EncodePipeline::EncodePipeline() : impl_(std::make_unique<Impl>()) {}
EncodePipeline::~EncodePipeline() { Stop(); }

bool EncodePipeline::Start(int width, int height, int fps, FrameCallback cb,
                           std::string* error_message) {
  gst_init(nullptr, nullptr);
  impl_->cb = std::move(cb);

  // Check for h264parse availability (non-fatal if absent, same logic as Argus backend).
  const bool has_h264parse =
      (gst_element_factory_find("h264parse") != nullptr);

  // Pipeline: NV12 appsrc → nvvidconv → I420 → x264enc → [h264parse →] appsink.
  // Using nvvidconv for colorspace conversion (VIC hardware block, free on Jetson).
  std::string pl;
  pl.reserve(1024);
  pl += "appsrc name=enc_src is-live=true format=time do-timestamp=true ";
  pl += "! video/x-raw,format=NV12";
  pl += ",width=" + std::to_string(width);
  pl += ",height=" + std::to_string(height);
  pl += ",framerate=" + std::to_string(fps) + "/1 ";
  pl += "! nvvidconv ";
  pl += "! video/x-raw,format=I420 ";
  pl += "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 ";
  pl += "vbv-buf-capacity=400 intra-refresh=true key-int-max=60 ";
  pl += "! video/x-h264,profile=baseline ";
  if (has_h264parse) {
    pl += "! h264parse config-interval=-1 ";
    pl += "! video/x-h264,stream-format=byte-stream,alignment=au ";
  }
  pl += "! appsink name=enc_sink emit-signals=true max-buffers=2 drop=true";

  spdlog::info("[EncodePipeline] Launching: {}", pl);

  GError* err      = nullptr;
  impl_->pipeline  = gst_parse_launch(pl.c_str(), &err);
  if (!impl_->pipeline) {
    const std::string msg = err ? std::string("Pipeline parse error: ") + err->message
                                : "Pipeline parse error (unknown)";
    if (err) g_error_free(err);
    if (error_message) *error_message = msg;
    spdlog::error("[EncodePipeline] {}", msg);
    return false;
  }
  if (err) g_error_free(err);

  impl_->appsrc  = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "enc_src");
  impl_->appsink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "enc_sink");
  if (!impl_->appsrc || !impl_->appsink) {
    const std::string msg = "Failed to find enc_src/enc_sink in EncodePipeline";
    if (error_message) *error_message = msg;
    spdlog::error("[EncodePipeline] {}", msg);
    impl_->Cleanup();
    return false;
  }

  GstAppSinkCallbacks cbs{};
  cbs.new_sample = &Impl::OnNewSample;
  gst_app_sink_set_callbacks(GST_APP_SINK(impl_->appsink), &cbs, impl_.get(), nullptr);

  if (gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    const std::string msg = "EncodePipeline failed to reach PLAYING";
    if (error_message) *error_message = msg;
    spdlog::error("[EncodePipeline] {}", msg);
    impl_->Cleanup();
    return false;
  }

  impl_->running    = true;
  impl_->bus_thread = std::thread([this] { impl_->BusLoop(); });
  spdlog::info("[EncodePipeline] Started {}x{}@{}fps", width, height, fps);
  return true;
}

void EncodePipeline::Stop() {
  if (!impl_->running.load() && impl_->pipeline == nullptr) return;
  impl_->running = false;
  if (impl_->appsrc) {
    gst_app_src_end_of_stream(GST_APP_SRC(impl_->appsrc));
  }
  if (impl_->bus_thread.joinable()) {
    impl_->bus_thread.join();
  }
  impl_->Cleanup();
  spdlog::info("[EncodePipeline] Stopped");
}

bool EncodePipeline::IsRunning() const { return impl_->running.load(); }

bool EncodePipeline::PushFrame(const uint8_t* nv12_host, int width, int height,
                               uint32_t rtp_ts) {
  if (!impl_->running.load() || !impl_->appsrc) return false;

  const gsize frame_size = static_cast<gsize>(width) * height * 3 / 2;
  GstBuffer* buf         = gst_buffer_new_allocate(nullptr, frame_size, nullptr);
  if (!buf) return false;

  GstMapInfo map;
  if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(buf);
    return false;
  }
  std::memcpy(map.data, nv12_host, frame_size);
  gst_buffer_unmap(buf, &map);

  // Use rtp_ts (µs) as PTS in GStreamer time units (nanoseconds).
  GST_BUFFER_PTS(buf) = static_cast<GstClockTime>(rtp_ts) * 1000ULL;

  const GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(impl_->appsrc), buf);
  if (ret != GST_FLOW_OK) {
    spdlog::debug("[EncodePipeline] gst_app_src_push_buffer: {}", static_cast<int>(ret));
    return false;
  }
  return true;
}

}  // namespace jrb::adapters::camera

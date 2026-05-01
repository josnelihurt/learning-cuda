#include "src/cpp_accelerator/adapters/camera/backends/nvidia_argus_backend.h"

#include <cstdlib>
#include <string>
#include <unistd.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

namespace {

bool IsNvargusAvailable(std::string* reason) {
  gst_init(nullptr, nullptr);

  GstElementFactory* factory = gst_element_factory_find("nvarguscamerasrc");
  if (!factory) {
    if (reason) {
      *reason = "GStreamer element 'nvarguscamerasrc' is not registered";
    }
    return false;
  }
  gst_object_unref(factory);

  if (access("/tmp/argus_socket", F_OK) != 0) {
    if (reason) {
      *reason = "Argus socket '/tmp/argus_socket' is not accessible inside container";
    }
    return false;
  }

  if (reason) {
    reason->clear();
  }
  return true;
}

bool HasGstElement(const char* element_name) {
  GstElementFactory* factory = gst_element_factory_find(element_name);
  if (!factory) {
    return false;
  }
  gst_object_unref(factory);
  return true;
}

constexpr GstClockTime kArgusPlayingWait = 10 * GST_SECOND;

// Best-effort: pull the next ERROR posted to the pipeline bus (Jetson pipelines
// often fail asynchronously during preroll; the reason is only on the bus).
std::string ConsumeNextBusError(GstElement* pipeline, GstClockTime timeout) {
  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) {
    return {};
  }

  GstMessage* msg =
      gst_bus_timed_pop_filtered(bus, timeout, GST_MESSAGE_ERROR);
  gst_object_unref(bus);
  if (!msg) {
    return {};
  }

  GError* err = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(msg, &err, &debug);
  std::string out;
  if (err && err->message) {
    out += err->message;
  }
  if (debug && debug[0] != '\0') {
    if (!out.empty()) {
      out += " — ";
    }
    out += debug;
  }
  if (err) {
    g_error_free(err);
  }
  if (debug) {
    g_free(debug);
  }
  gst_message_unref(msg);
  return out;
}

void CleanupFailedPipeline(GstElement** pipeline, GstElement** appsink) {
  if (appsink && *appsink) {
    gst_object_unref(*appsink);
    *appsink = nullptr;
  }
  if (pipeline && *pipeline) {
    gst_element_set_state(*pipeline, GST_STATE_NULL);
    gst_object_unref(*pipeline);
    *pipeline = nullptr;
  }
}

bool ProbeSensor(int sensor_id) {
  char pipeline_str[256];
  snprintf(pipeline_str, sizeof(pipeline_str),
           "nvarguscamerasrc sensor-id=%d num-buffers=1 ! fakesink", sensor_id);

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_str, &err);
  if (!pipeline) {
    if (err) g_error_free(err);
    return false;
  }
  if (err) {
    g_error_free(err);
    gst_object_unref(pipeline);
    return false;
  }

  GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return false;
  }

  GstBus* bus = gst_element_get_bus(pipeline);
  bool success = false;
  bool done = false;

  while (!done) {
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus,
        5 * GST_SECOND,
        static_cast<GstMessageType>(GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR |
                                    GST_MESSAGE_EOS));
    if (!msg) break;

    GstMessageType type = GST_MESSAGE_TYPE(msg);
    if (type == GST_MESSAGE_STATE_CHANGED) {
      GstState old_state, new_state, pending;
      gst_message_parse_state_changed(msg, &old_state, &new_state, &pending);
      if (GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline) &&
          new_state == GST_STATE_PLAYING) {
        success = true;
        done = true;
      }
    } else if (type == GST_MESSAGE_EOS) {
      success = true;
      done = true;
    } else if (type == GST_MESSAGE_ERROR) {
      GError* gerr = nullptr;
      gchar* debug = nullptr;
      gst_message_parse_error(msg, &gerr, &debug);
      if (gerr) g_error_free(gerr);
      if (debug) g_free(debug);
      done = true;
    }
    gst_message_unref(msg);
  }

  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return success;
}

}  // namespace

struct NvidiaArgusBackend::Impl {
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
        spdlog::warn("[NvidiaArgusBackend] Frame callback threw: {}", e.what());
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
        spdlog::error("[NvidiaArgusBackend] Pipeline error: {} ({})",
                      err ? err->message : "unknown",
                      debug ? debug : "");
        if (err) g_error_free(err);
        if (debug) g_free(debug);
        running = false;
      } else if (type == GST_MESSAGE_EOS) {
        spdlog::info("[NvidiaArgusBackend] Pipeline EOS");
        running = false;
      }
      gst_message_unref(msg);
    }
    gst_object_unref(bus);
  }
};

NvidiaArgusBackend::NvidiaArgusBackend() : impl_(std::make_unique<Impl>()) {}

NvidiaArgusBackend::~NvidiaArgusBackend() {
  Stop();
}

bool NvidiaArgusBackend::IsAvailable() const {
  std::string reason;
  const bool available = IsNvargusAvailable(&reason);
  if (!available) {
    spdlog::warn("[NvidiaArgusBackend] Backend unavailable: {}", reason);
  }
  return available;
}

std::vector<cuda_learning::RemoteCameraInfo> NvidiaArgusBackend::DetectCameras(
    const std::vector<int>& sensor_ids) {
  std::vector<cuda_learning::RemoteCameraInfo> result;

  std::string availability_reason;
  if (!IsNvargusAvailable(&availability_reason)) {
    spdlog::warn("[NvidiaArgusBackend] Skipping detection: {}", availability_reason);
    return result;
  }

  for (int sensor_id : sensor_ids) {
    try {
      spdlog::info("[NvidiaArgusBackend] Probing sensor-id={}", sensor_id);
      if (!ProbeSensor(sensor_id)) {
        spdlog::warn("[NvidiaArgusBackend] sensor-id={} probe failed", sensor_id);
        continue;
      }

      cuda_learning::RemoteCameraInfo info;
      info.set_sensor_id(sensor_id);
      const std::string display_name =
          "Argus: CAM" + std::to_string(sensor_id) +
          " (sensor-id=" + std::to_string(sensor_id) + ")";
      info.set_display_name(display_name);
      info.set_model("");

      auto* mode = info.add_modes();
      mode->set_width(1920);
      mode->set_height(1080);
      mode->set_fps(60.0);

      spdlog::info("[NvidiaArgusBackend] sensor-id={} detected: {}", sensor_id, display_name);
      result.push_back(std::move(info));
    } catch (const std::exception& e) {
      spdlog::warn("[NvidiaArgusBackend] Exception probing sensor-id={}: {}", sensor_id, e.what());
    } catch (...) {
      spdlog::warn("[NvidiaArgusBackend] Unknown exception probing sensor-id={}", sensor_id);
    }
  }

  return result;
}

void NvidiaArgusBackend::SetFrameCallback(FrameCallback cb) {
  impl_->cb = std::move(cb);
}

bool NvidiaArgusBackend::Start(int sensor_id, int width, int height, int fps,
                               std::string* error_message) {
  gst_init(nullptr, nullptr);

  constexpr int kDefaultWidth = 1920;
  constexpr int kDefaultHeight = 1080;
  constexpr int kDefaultFps = 30;
  const int effective_width = width > 0 ? width : kDefaultWidth;
  const int effective_height = height > 0 ? height : kDefaultHeight;
  const int effective_fps = fps > 0 ? fps : kDefaultFps;

  if (width <= 0 || height <= 0 || fps <= 0) {
    spdlog::warn(
        "[NvidiaArgusBackend] Invalid camera params {}x{}@{}fps for sensor-id={}; "
        "using defaults {}x{}@{}fps",
        width, height, fps, sensor_id, effective_width, effective_height, effective_fps);
  }

  const bool has_h264parse = HasGstElement("h264parse");
  if (!has_h264parse) {
    spdlog::warn(
        "[NvidiaArgusBackend] GStreamer element 'h264parse' not found; "
        "continuing without parser in Argus pipeline");
  }

  // Orin Nano has no NVENC: the nvv4l2h264enc plugin is registered in the L4T
  // image, but it opens /dev/v4l2-nvenc, which the kernel never creates on
  // SKUs without an encoder. So check the device node, not just the factory.
  const bool nvenc_plugin = HasGstElement("nvv4l2h264enc");
  const bool nvenc_device = (access("/dev/v4l2-nvenc", F_OK) == 0);
  const bool has_nvenc = nvenc_plugin && nvenc_device;
  const bool has_x264 = HasGstElement("x264enc");
  if (!has_nvenc) {
    spdlog::warn(
        "[NvidiaArgusBackend] Hardware H.264 encoder unavailable "
        "(plugin_registered={}, device_present={}); falling back to software "
        "'x264enc' (expected on Orin Nano: no NVENC hardware)",
        nvenc_plugin, nvenc_device);
  }
  if (!has_nvenc && !has_x264) {
    const std::string msg =
        "No H.264 encoder available: neither nvv4l2h264enc nor x264enc are "
        "registered with GStreamer";
    if (error_message) *error_message = msg;
    spdlog::error("[NvidiaArgusBackend] {}", msg);
    return false;
  }

  // Hardware path keeps NVMM all the way to the encoder; software path needs a
  // download to system memory (I420) before x264enc.
  const char* encoder_segment =
      has_nvenc
          ? "nvvidconv ! "
            "nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! "
          : "nvvidconv ! video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency speed-preset=ultrafast "
            "bitrate=2000 key-int-max=30 ! "
            "video/x-h264,profile=baseline ! ";

  // Pin Annex-B byte-stream / AU alignment after the parser. Without this,
  // h264parse may negotiate to stream-format=avc (length-prefixed NALs),
  // which the downstream libavcodec decoder in live_video_processor cannot
  // parse — producing "No start code is found" / "Invalid NAL unit 0".
  const char* parse_segment =
      has_h264parse
          ? "h264parse config-interval=-1 ! "
            "video/x-h264,stream-format=byte-stream,alignment=au ! "
          : "";

  char pipeline_str[896];
  snprintf(pipeline_str, sizeof(pipeline_str),
           "nvarguscamerasrc sensor-id=%d ! "
           "video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1,format=NV12 ! "
           "%s"
           "%s"
           "appsink name=sink emit-signals=true max-buffers=2 drop=true",
           sensor_id, effective_width, effective_height, effective_fps,
           encoder_segment,
           parse_segment);

  spdlog::info("[NvidiaArgusBackend] Launching pipeline: {}", pipeline_str);

  GError* err = nullptr;
  impl_->pipeline = gst_parse_launch(pipeline_str, &err);
  if (!impl_->pipeline) {
    const std::string msg = err ? std::string("Failed to parse pipeline: ") + err->message
                                : "Failed to parse pipeline (unknown error)";
    if (err) g_error_free(err);
    if (error_message) *error_message = msg;
    spdlog::error("[NvidiaArgusBackend] {}", msg);
    return false;
  }
  if (err) {
    g_error_free(err);
  }

  impl_->appsink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "sink");
  if (!impl_->appsink) {
    if (error_message) *error_message = "Failed to find appsink element";
    spdlog::error("[NvidiaArgusBackend] appsink element not found");
    gst_object_unref(impl_->pipeline);
    impl_->pipeline = nullptr;
    return false;
  }

  GstAppSinkCallbacks callbacks{};
  callbacks.new_sample = &Impl::OnNewSample;
  gst_app_sink_set_callbacks(GST_APP_SINK(impl_->appsink), &callbacks, impl_.get(), nullptr);

  GstStateChangeReturn ret = gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    std::string bus_detail = ConsumeNextBusError(impl_->pipeline, GST_SECOND);
    std::string msg = "Failed to set pipeline to PLAYING state";
    if (!bus_detail.empty()) {
      msg += ": ";
      msg += bus_detail;
    }
    if (error_message) {
      *error_message = msg;
    }
    spdlog::error("[NvidiaArgusBackend] {} (sensor-id={})", msg, sensor_id);
    CleanupFailedPipeline(&impl_->pipeline, &impl_->appsink);
    return false;
  }

  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_VOID_PENDING;
  const GstStateChangeReturn waited =
      gst_element_get_state(impl_->pipeline, &state, &pending, kArgusPlayingWait);
  if (waited == GST_STATE_CHANGE_FAILURE || state != GST_STATE_PLAYING) {
    std::string bus_detail = ConsumeNextBusError(impl_->pipeline, 2 * GST_SECOND);
    std::string msg = "Pipeline did not reach PLAYING";
    if (waited == GST_STATE_CHANGE_FAILURE) {
      msg += " (get_state failure)";
    } else {
      msg += " (state=" + std::string(gst_element_state_get_name(state)) +
             ", pending=" + std::string(gst_element_state_get_name(pending)) + ")";
    }
    if (!bus_detail.empty()) {
      msg += ": ";
      msg += bus_detail;
    }
    if (error_message) {
      *error_message = msg;
    }
    spdlog::error("[NvidiaArgusBackend] {} (sensor-id={})", msg, sensor_id);
    CleanupFailedPipeline(&impl_->pipeline, &impl_->appsink);
    return false;
  }

  impl_->running = true;
  impl_->bus_thread = std::thread([this]() { impl_->BusLoop(); });

  spdlog::info(
      "[NvidiaArgusBackend] Started camera sensor-id={} {}x{}@{}fps",
      sensor_id, effective_width, effective_height, effective_fps);
  return true;
}

void NvidiaArgusBackend::Stop() {
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
    spdlog::info("[NvidiaArgusBackend] Pipeline stopped and released");
  }
}

bool NvidiaArgusBackend::IsRunning() const {
  return impl_->running.load();
}

std::string NvidiaArgusBackend::GetBackendName() const {
  return "Argus";
}

}  // namespace jrb::adapters::camera

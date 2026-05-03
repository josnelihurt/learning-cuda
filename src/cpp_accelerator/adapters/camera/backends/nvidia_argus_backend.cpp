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

// IMX477 full-sensor mode dimensions and fps.
constexpr int kSensorWidth  = 4056;
constexpr int kSensorHeight = 3040;
constexpr int kSensorFps    = 15;

// Default encode resolution when the caller passes 0×0.
constexpr int kDefaultEncodeWidth  = 1280;
constexpr int kDefaultEncodeHeight = 720;

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
  GstElement* proc_sink{nullptr};   // continuous H.264 callbacks (encode branch)
  GstElement* still_sink{nullptr};  // pull-on-demand for full-res NV12 stills
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

  void Cleanup() {
    if (pipeline) {
      gst_element_set_state(pipeline, GST_STATE_NULL);
      if (proc_sink) {
        gst_object_unref(proc_sink);
        proc_sink = nullptr;
      }
      if (still_sink) {
        gst_object_unref(still_sink);
        still_sink = nullptr;
      }
      gst_object_unref(pipeline);
      pipeline = nullptr;
    }
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

      // Full-sensor readout — used for high-res still capture; encode branch
      // hardware-downscales to the requested streaming resolution.
      auto* mode_full = info.add_modes();
      mode_full->set_width(kSensorWidth);
      mode_full->set_height(kSensorHeight);
      mode_full->set_fps(static_cast<double>(kSensorFps));

      // Streaming mode: 720p encode at sensor fps.
      auto* mode_stream = info.add_modes();
      mode_stream->set_width(kDefaultEncodeWidth);
      mode_stream->set_height(kDefaultEncodeHeight);
      mode_stream->set_fps(static_cast<double>(kSensorFps));

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

  // Encode-branch resolution: what gets H.264-encoded and delivered to subscribers.
  // The Argus source always captures at full sensor resolution (4056×3040@15fps).
  const int encode_w = width  > 0 ? width  : kDefaultEncodeWidth;
  const int encode_h = height > 0 ? height : kDefaultEncodeHeight;

  if (width <= 0 || height <= 0) {
    spdlog::warn(
        "[NvidiaArgusBackend] Invalid encode params {}x{} for sensor-id={}; "
        "using defaults {}x{}",
        width, height, sensor_id, encode_w, encode_h);
  }
  // fps is informational only; the sensor drives the clock at kSensorFps.
  (void)fps;

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

  // AWB workaround for Arducam IMX477: stock ISP tuning + wbmode=auto produces
  // a strong cyan cast on mixed indoor/window-light scenes. Lock to a fixed
  // preset. Override per-deploy with NVARGUS_WBMODE; default 4 (warm-fluorescent).
  const char* wbmode_env = std::getenv("NVARGUS_WBMODE");
  const int wbmode = wbmode_env ? std::atoi(wbmode_env) : 4;

  // Build pipeline string.
  //
  // Architecture:
  //   nvarguscamerasrc (4056×3040 @ 15fps, full sensor readout)
  //     → tee
  //         still_sink branch : raw NV12 frames, pull-on-demand for BMP stills
  //         encode branch     : nvvidconv downscale → H.264 encoder → proc_sink
  //
  // The still_sink branch keeps the full 4056×3040 NV12 frame in a one-buffer
  // leaky queue.  GrabStillFrame() pulls from it on demand (~67 ms wait at 15fps).
  // The encode branch downscales to encode_w×encode_h before the H.264 encoder,
  // so x264enc CPU load stays at ~15% (same as the previous 720p pipeline).
  std::string pipeline;
  pipeline.reserve(2048);

  // Source — always full sensor resolution at sensor native fps.
  pipeline += "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id);
  pipeline += " wbmode=" + std::to_string(wbmode) + " ! ";
  pipeline += "video/x-raw(memory:NVMM)";
  pipeline += ",width="     + std::to_string(kSensorWidth);
  pipeline += ",height="    + std::to_string(kSensorHeight);
  pipeline += ",framerate=" + std::to_string(kSensorFps) + "/1";
  pipeline += ",format=NV12 ! ";
  pipeline += "tee name=cam_tee ";

  // Still-capture branch: nvvidconv normalizes the NVMM pitch to stride=width in
  // system memory so gst_buffer_map gives linear NV12 without alignment padding.
  // nvvidconv uses the VIC hardware block (effectively free on Jetson).
  // The leaky queue (downstream, max 1 buffer) keeps only the latest frame so
  // GrabStillFrame always returns a fresh ~67 ms-old frame regardless of pull cadence.
  pipeline += "cam_tee. ! queue leaky=2 max-size-buffers=1 ! ";
  pipeline += "nvvidconv ! ";
  pipeline += "video/x-raw,width=" + std::to_string(kSensorWidth);
  pipeline += ",height=" + std::to_string(kSensorHeight);
  pipeline += ",format=NV12 ! ";
  pipeline += "appsink name=still_sink emit-signals=false sync=false max-buffers=1 drop=true ";

  // Encode branch: hardware downscale then H.264 encode.
  pipeline += "cam_tee. ! ";
  if (has_nvenc) {
    // Hardware path: nvvidconv keeps data in NVMM through the encoder.
    pipeline += "nvvidconv ! ";
    pipeline += "video/x-raw(memory:NVMM)";
    pipeline += ",width="  + std::to_string(encode_w);
    pipeline += ",height=" + std::to_string(encode_h);
    pipeline += ",format=NV12 ! ";
    pipeline += "nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! ";
  } else {
    // Software path: nvvidconv downscales and converts NVMM→I420 for x264enc.
    // Rolling intra-refresh avoids large keyframe bursts that overrun the
    // WebRTC outbound queue (see backends/README.md §13.3).
    pipeline += "nvvidconv ! ";
    pipeline += "video/x-raw";
    pipeline += ",width="  + std::to_string(encode_w);
    pipeline += ",height=" + std::to_string(encode_h);
    pipeline += ",format=I420 ! ";
    pipeline += "x264enc tune=zerolatency speed-preset=ultrafast ";
    pipeline += "bitrate=2000 vbv-buf-capacity=400 ";
    pipeline += "intra-refresh=true key-int-max=60 ! ";
    pipeline += "video/x-h264,profile=baseline ! ";
  }

  // Pin Annex-B byte-stream / AU alignment after the parser. Without this,
  // h264parse may negotiate to stream-format=avc (length-prefixed NALs),
  // which the downstream libavcodec decoder rejects — "No start code is found".
  if (has_h264parse) {
    pipeline += "h264parse config-interval=-1 ! ";
    pipeline += "video/x-h264,stream-format=byte-stream,alignment=au ! ";
  }
  pipeline += "appsink name=proc_sink emit-signals=true max-buffers=2 drop=true";

  spdlog::info("[NvidiaArgusBackend] Launching pipeline: {}", pipeline);

  GError* err = nullptr;
  impl_->pipeline = gst_parse_launch(pipeline.c_str(), &err);
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

  impl_->proc_sink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "proc_sink");
  if (!impl_->proc_sink) {
    if (error_message) *error_message = "Failed to find proc_sink element";
    spdlog::error("[NvidiaArgusBackend] proc_sink element not found");
    impl_->Cleanup();
    return false;
  }

  impl_->still_sink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "still_sink");
  if (!impl_->still_sink) {
    // Non-fatal: still capture won't work but streaming can continue.
    spdlog::warn("[NvidiaArgusBackend] still_sink element not found; still capture unavailable");
  }

  GstAppSinkCallbacks callbacks{};
  callbacks.new_sample = &Impl::OnNewSample;
  gst_app_sink_set_callbacks(GST_APP_SINK(impl_->proc_sink), &callbacks, impl_.get(), nullptr);

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
    impl_->Cleanup();
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
    impl_->Cleanup();
    return false;
  }

  impl_->running = true;
  impl_->bus_thread = std::thread([this]() { impl_->BusLoop(); });

  spdlog::info(
      "[NvidiaArgusBackend] Started sensor-id={} full-sensor {}x{}@{}fps, "
      "encode branch {}x{}",
      sensor_id, kSensorWidth, kSensorHeight, kSensorFps, encode_w, encode_h);
  return true;
}

void NvidiaArgusBackend::Stop() {
  if (!impl_->running.load() && impl_->pipeline == nullptr) return;

  impl_->running = false;

  if (impl_->bus_thread.joinable()) {
    impl_->bus_thread.join();
  }

  impl_->Cleanup();
  spdlog::info("[NvidiaArgusBackend] Pipeline stopped and released");
}

bool NvidiaArgusBackend::IsRunning() const {
  return impl_->running.load();
}

std::string NvidiaArgusBackend::GetBackendName() const {
  return "Argus";
}

rtc::binary NvidiaArgusBackend::GrabStillFrame(int* out_width, int* out_height) {
  if (!impl_->running.load() || !impl_->still_sink) {
    spdlog::warn("[NvidiaArgusBackend] GrabStillFrame: pipeline not running or still_sink absent");
    return {};
  }

  // Wait up to 500 ms for the leaky queue to have a fresh frame (at 15fps, a
  // new frame arrives every ~67 ms so 500 ms is more than enough).
  GstSample* sample = gst_app_sink_try_pull_sample(
      GST_APP_SINK(impl_->still_sink), 500 * GST_MSECOND);
  if (!sample) {
    spdlog::warn("[NvidiaArgusBackend] GrabStillFrame: timed out waiting for still frame");
    return {};
  }

  if (out_width || out_height) {
    GstCaps* caps = gst_sample_get_caps(sample);
    if (caps) {
      const GstStructure* s = gst_caps_get_structure(caps, 0);
      int w = 0, h = 0;
      gst_structure_get_int(s, "width", &w);
      gst_structure_get_int(s, "height", &h);
      if (out_width)  *out_width  = w;
      if (out_height) *out_height = h;
    }
  }

  GstBuffer* buf = gst_sample_get_buffer(sample);
  rtc::binary data;
  if (buf) {
    // The still_sink branch goes through nvvidconv to system memory, so
    // map.data is linear NV12 with stride=width — no alignment padding.
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
      data.assign(reinterpret_cast<const std::byte*>(map.data),
                  reinterpret_cast<const std::byte*>(map.data) + map.size);
      gst_buffer_unmap(buf, &map);
    }
  }
  gst_sample_unref(sample);
  return data;
}

}  // namespace jrb::adapters::camera

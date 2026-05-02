// hello-world-nvidia-argus: validates NVIDIA Argus camera framework on Jetson.
// Probes sensors via nvarguscamerasrc, then captures H.264 frames from each.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

static const int kMaxSensorId = 9;
static const int kDefaultFrameCount = 10;
static const int kDefaultWidth = 1920;
static const int kDefaultHeight = 1080;
static const int kDefaultFps = 30;

struct DetectedCamera {
  int sensor_id;
  std::string display_name;
};

struct FrameStats {
  int frame_index;
  size_t size_bytes;
  uint64_t pts_ns;
};

static bool IsArgusAvailable() {
  FILE* fp = popen("gst-inspect-1.0 nvarguscamerasrc 2>&1", "r");
  if (!fp) return false;
  char buf[256];
  bool found = false;
  while (fgets(buf, sizeof(buf), fp)) {
    std::string line(buf);
    if (line.find("NvArgusCameraSrc") != std::string::npos) {
      found = true;
      break;
    }
  }
  pclose(fp);
  return found;
}

static std::string FindBestH264Encoder() {
  // Prefer nvv4l2h264enc (Jetson hardware encoder), fall back to x264enc.
  const char* candidates[] = {"nvv4l2h264enc", "x264enc", "openh264enc"};
  for (const char* name : candidates) {
    std::string cmd = "gst-inspect-1.0 ";
    cmd += name;
    cmd += " >/dev/null 2>&1";
    if (system(cmd.c_str()) == 0) {
      return name;
    }
  }
  return "";
}

// Builds the encoder portion of the pipeline depending on which element is available.
// nvv4l2h264enc accepts NVMM directly; x264enc/openh264enc need system memory.
static std::string BuildEncoderPipeline(const std::string& encoder, int width, int height) {
  if (encoder == "nvv4l2h264enc") {
    return "nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! "
           "h264parse config-interval=-1 ! "
           "appsink name=sink emit-signals=true max-buffers=2 drop=true";
  }
  // Software encoders: nvvidconv must output system memory.
  char caps[128];
  snprintf(caps, sizeof(caps),
           "video/x-raw,width=%d,height=%d ! ", width, height);

  if (encoder == "x264enc") {
    return std::string(caps) +
           "x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
           "video/x-h264,stream-format=byte-stream,alignment=au ! "
           "h264parse config-interval=-1 ! "
           "appsink name=sink emit-signals=true max-buffers=2 drop=true";
  }
  // openh264enc fallback
  return std::string(caps) +
         "openh264enc ! "
         "video/x-h264,stream-format=byte-stream,alignment=au ! "
         "h264parse config-interval=-1 ! "
         "appsink name=sink emit-signals=true max-buffers=2 drop=true";
}

// Probes a single sensor by launching a minimal pipeline.
// num-buffers=1 causes EOS quickly, which we treat as success.
static bool ProbeSensor(int sensor_id) {
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
      fprintf(stderr, "  [probe] sensor-id=%d error: %s (%s)\n",
              sensor_id,
              gerr ? gerr->message : "unknown",
              debug ? debug : "");
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

static std::vector<FrameStats> CaptureFrames(int sensor_id, int width, int height,
                                             int fps, int max_frames,
                                             const std::string& encoder) {
  std::string encoder_part = BuildEncoderPipeline(encoder, width, height);

  char pipeline_str[1024];
  snprintf(pipeline_str, sizeof(pipeline_str),
           "nvarguscamerasrc sensor-id=%d ! "
           "video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1,format=NV12 ! "
           "nvvidconv ! "
           "%s",
           sensor_id, width, height, fps, encoder_part.c_str());

  printf("  [stream] pipeline: %s\n", pipeline_str);

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_str, &err);
  if (!pipeline) {
    fprintf(stderr, "  [stream] failed to parse pipeline: %s\n",
            err ? err->message : "unknown");
    if (err) g_error_free(err);
    return {};
  }
  if (err) g_error_free(err);

  GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
  if (!appsink) {
    fprintf(stderr, "  [stream] appsink element not found\n");
    gst_object_unref(pipeline);
    return {};
  }

  GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    fprintf(stderr, "  [stream] failed to start pipeline\n");
    gst_object_unref(appsink);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return {};
  }

  // Wait for the pipeline to reach PLAYING. Both SUCCESS and ASYNC are fine;
  // only FAILURE means the pipeline could not start.
  ret = gst_element_get_state(pipeline, nullptr, nullptr, 10 * GST_SECOND);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    fprintf(stderr, "  [stream] pipeline failed to reach PLAYING\n");
    gst_object_unref(appsink);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return {};
  }

  std::vector<FrameStats> frames;
  frames.reserve(max_frames);

  for (int i = 0; i < max_frames; ++i) {
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 5 * GST_SECOND);
    if (!sample) {
      fprintf(stderr, "  [stream] timeout pulling sample %d/%d\n", i + 1, max_frames);
      break;
    }

    GstBuffer* buf = gst_sample_get_buffer(sample);
    if (!buf) {
      gst_sample_unref(sample);
      continue;
    }

    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
      FrameStats stats;
      stats.frame_index = i;
      stats.size_bytes = map.size;
      stats.pts_ns = GST_BUFFER_PTS_IS_VALID(buf) ? GST_BUFFER_PTS(buf) : 0;
      frames.push_back(stats);
      gst_buffer_unmap(buf, &map);
    }

    gst_sample_unref(sample);
  }

  gst_object_unref(appsink);
  gst_element_set_state(pipeline, GST_STATE_NULL);

  GstBus* bus = gst_element_get_bus(pipeline);
  gst_bus_set_flushing(bus, true);
  gst_object_unref(bus);

  gst_object_unref(pipeline);
  return frames;
}

static void PrintUsage(const char* prog) {
  fprintf(stderr,
          "Usage: %s [options]\n"
          "Options:\n"
          "  --max-sensor <N>   Max sensor ID to probe (default %d)\n"
          "  --frames <N>       Frames to capture per camera (default %d)\n"
          "  --width <N>        Capture width (default %d)\n"
          "  --height <N>       Capture height (default %d)\n"
          "  --fps <N>          Capture framerate (default %d)\n"
          "  --detect-only      Only detect cameras, don't stream\n"
          "  --help             Show this help\n",
          prog, kMaxSensorId, kDefaultFrameCount, kDefaultWidth, kDefaultHeight,
          kDefaultFps);
}

int main(int argc, char* argv[]) {
  int max_sensor = kMaxSensorId;
  int frame_count = kDefaultFrameCount;
  int width = kDefaultWidth;
  int height = kDefaultHeight;
  int fps = kDefaultFps;
  bool detect_only = false;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--max-sensor") == 0 && i + 1 < argc) {
      max_sensor = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
      frame_count = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
      width = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
      height = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
      fps = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--detect-only") == 0) {
      detect_only = true;
    } else if (strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      PrintUsage(argv[0]);
      return 1;
    }
  }

  printf("=== hello-world-nvidia-argus ===\n\n");

  gst_init(nullptr, nullptr);
  printf("[init] GStreamer initialized\n");

  if (!IsArgusAvailable()) {
    fprintf(stderr, "[error] nvarguscamerasrc not found. Is this a Jetson with Argus?\n");
    return 1;
  }
  printf("[init] nvarguscamerasrc is available\n");

  std::string encoder = FindBestH264Encoder();
  if (encoder.empty()) {
    fprintf(stderr, "[error] No H.264 encoder found (tried nvv4l2h264enc, x264enc, openh264enc)\n");
    return 1;
  }
  printf("[init] H.264 encoder: %s\n\n", encoder.c_str());

  // Phase 1: Detect cameras.
  printf("--- Phase 1: Camera Detection (sensor 0..%d) ---\n", max_sensor);
  std::vector<DetectedCamera> cameras;
  for (int id = 0; id <= max_sensor; ++id) {
    printf("  Probing sensor-id=%d ... ", id);
    fflush(stdout);
    if (ProbeSensor(id)) {
      DetectedCamera cam;
      cam.sensor_id = id;
      cam.display_name = "CAM" + std::to_string(id) + " (sensor-id=" +
                          std::to_string(id) + ")";
      cameras.push_back(std::move(cam));
      printf("DETECTED\n");
    } else {
      printf("not found\n");
    }
  }

  if (cameras.empty()) {
    fprintf(stderr, "\n[error] No cameras detected\n");
    return 1;
  }

  printf("\nDetected %zu camera(s):\n", cameras.size());
  for (const auto& cam : cameras) {
    printf("  - %s: %dx%d@%dfps\n", cam.display_name.c_str(), width, height, fps);
  }

  if (detect_only) {
    printf("\n--detect-only: skipping stream test\n");
    printf("\n=== PASSED ===\n");
    return 0;
  }

  // Phase 2: Stream and capture frames from each camera.
  printf("\n--- Phase 2: Frame Capture ---\n");
  int total_failures = 0;

  for (const auto& cam : cameras) {
    printf("\n[cam] %s — capturing %d frames at %dx%d@%dfps\n",
           cam.display_name.c_str(), frame_count, width, height, fps);

    auto frames = CaptureFrames(cam.sensor_id, width, height, fps, frame_count, encoder);

    if (frames.empty()) {
      fprintf(stderr, "  [FAIL] No frames captured from %s\n", cam.display_name.c_str());
      total_failures++;
      continue;
    }

    printf("  Captured %zu/%d frames:\n", frames.size(), frame_count);
    for (const auto& f : frames) {
      printf("    frame #%d: %zu bytes, pts=%lu ns\n",
             f.frame_index, f.size_bytes, f.pts_ns);
    }

    if (static_cast<int>(frames.size()) < frame_count) {
      fprintf(stderr, "  [WARN] Expected %d frames, got %zu\n",
              frame_count, frames.size());
    }
  }

  printf("\n--- Summary ---\n");
  printf("Cameras detected: %zu\n", cameras.size());
  printf("Stream failures:  %d\n", total_failures);

  if (total_failures > 0) {
    printf("\n=== FAILED ===\n");
    return 1;
  }

  printf("\n=== PASSED ===\n");
  return 0;
}

#include "src/cpp_accelerator/adapters/camera/camera_detector.h"

#include <cstdio>
#include <string>
#include <vector>

#include <gst/gst.h>
#include <spdlog/spdlog.h>

namespace jrb::adapters::camera {

namespace {

bool IsNvargusAvailable() {
  FILE* fp = popen("gst-inspect-1.0 nvarguscamerasrc 2>&1", "r");
  if (!fp) return false;
  char buf[256];
  bool found = false;
  while (fgets(buf, sizeof(buf), fp)) {
    std::string line(buf);
    if (line.find("nvarguscamerasrc") != std::string::npos &&
        line.find("Factory Details") != std::string::npos) {
      found = true;
      break;
    }
  }
  pclose(fp);
  return found;
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
  GstMessage* msg = gst_bus_timed_pop_filtered(
      bus,
      3 * GST_SECOND,
      static_cast<GstMessageType>(GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  if (msg) {
    GstMessageType type = GST_MESSAGE_TYPE(msg);
    if (type == GST_MESSAGE_STATE_CHANGED) {
      GstState old_state, new_state, pending;
      gst_message_parse_state_changed(msg, &old_state, &new_state, &pending);
      if (GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline) &&
          new_state == GST_STATE_PLAYING) {
        success = true;
      }
    } else if (type == GST_MESSAGE_ERROR) {
      GError* gerr = nullptr;
      gchar* debug = nullptr;
      gst_message_parse_error(msg, &gerr, &debug);
      if (gerr) g_error_free(gerr);
      if (debug) g_free(debug);
    }
    gst_message_unref(msg);
  } else {
    GstState state;
    GstStateChangeReturn sr = gst_element_get_state(pipeline, &state, nullptr, 100 * GST_MSECOND);
    if (sr != GST_STATE_CHANGE_FAILURE && state == GST_STATE_PLAYING) {
      success = true;
    }
  }

  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return success;
}

}  // namespace

std::vector<cuda_learning::RemoteCameraInfo> DetectCameras(
    const std::vector<int>& sensor_ids) {
  std::vector<cuda_learning::RemoteCameraInfo> result;

  gst_init(nullptr, nullptr);

  if (!IsNvargusAvailable()) {
    spdlog::info("[CameraDetector] nvarguscamerasrc not found — skipping camera detection");
    return result;
  }

  for (int sensor_id : sensor_ids) {
    try {
      if (!ProbeSensor(sensor_id)) {
        spdlog::debug("[CameraDetector] sensor-id={} not detected (probe failed)", sensor_id);
        continue;
      }

      cuda_learning::RemoteCameraInfo info;
      info.set_sensor_id(sensor_id);
      const std::string display_name =
          "CAM" + std::to_string(sensor_id) +
          " (sensor-id=" + std::to_string(sensor_id) + ")";
      info.set_display_name(display_name);
      info.set_model("");

      auto* mode = info.add_modes();
      mode->set_width(1920);
      mode->set_height(1080);
      mode->set_fps(60.0);

      spdlog::info("[CameraDetector] sensor-id={} detected: {}", sensor_id, display_name);
      result.push_back(std::move(info));
    } catch (const std::exception& e) {
      spdlog::warn("[CameraDetector] Exception probing sensor-id={}: {}", sensor_id, e.what());
    } catch (...) {
      spdlog::warn("[CameraDetector] Unknown exception probing sensor-id={}", sensor_id);
    }
  }

  return result;
}

}  // namespace jrb::adapters::camera

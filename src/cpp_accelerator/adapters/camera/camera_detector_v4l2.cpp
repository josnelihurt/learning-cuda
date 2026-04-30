#include "src/cpp_accelerator/adapters/camera/camera_detector.h"

#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/videodev2.h>
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

std::vector<cuda_learning::RemoteCameraInfo> DetectCameras(
    const std::vector<int>& sensor_ids) {
  std::vector<cuda_learning::RemoteCameraInfo> result;

  for (int sensor_id : sensor_ids) {
    const std::string device_path = "/dev/video" + std::to_string(sensor_id);

    const std::string card_name = QueryV4L2DeviceName(device_path);
    if (card_name.empty()) {
      spdlog::debug("[CameraDetector] {} not a V4L2 capture device — skipping", device_path);
      continue;
    }

    cuda_learning::RemoteCameraInfo info;
    info.set_sensor_id(sensor_id);
    const std::string display_name =
        card_name + " (" + device_path + ")";
    info.set_display_name(display_name);
    info.set_model(card_name);

    auto* mode = info.add_modes();
    mode->set_width(1920);
    mode->set_height(1080);
    mode->set_fps(30.0);

    spdlog::info("[CameraDetector] Detected V4L2 camera at {}: {}", device_path, display_name);
    result.push_back(std::move(info));
  }

  return result;
}

}  // namespace jrb::adapters::camera

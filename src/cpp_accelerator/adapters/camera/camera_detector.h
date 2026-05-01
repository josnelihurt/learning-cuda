#pragma once

#include <string>
#include <vector>

#include "proto/_virtual_imports/accelerator_control_proto/accelerator_control.pb.h"

namespace jrb::adapters::camera {

// Probes each sensor ID via GStreamer nvarguscamerasrc and returns a
// RemoteCameraInfo for each that responds successfully.
// On non-Jetson platforms (no nvarguscamerasrc) returns an empty list.
std::vector<cuda_learning::RemoteCameraInfo> DetectCameras(
    const std::vector<int>& sensor_ids);

}  // namespace jrb::adapters::camera

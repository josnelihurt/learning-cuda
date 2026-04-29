#pragma once

#include <string>
#include <string_view>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"

namespace jrb::adapters::webrtc::protocol {

constexpr std::string_view kGoVideoSessionPrefix = "go-video-";
constexpr uint32_t kProcessedVideoBitrate = 2'500'000;
constexpr const char* kProcessedVideoTrackLabel = "processed-video";
constexpr const char* kStatsChannelLabel = "cpp-processor-stats";
constexpr const char* kControlChannelLabel = "control";

bool IsGoVideoSession(const std::string& value);
bool ShouldRegisterSessionChannel(const std::string& session_id, const std::string& label);
std::string AcceleratorTypeLabel(cuda_learning::AcceleratorType type);

}  // namespace jrb::adapters::webrtc::protocol

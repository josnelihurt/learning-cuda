#pragma once

#include <cstdint>
#include <string_view>

namespace jrb::adapters::webrtc {

// Sessions whose ID begins with this prefix originate from the Go side and
// route their responses through the gRPC control plane rather than back over
// the data channel — see ShouldRegisterSessionChannel().
constexpr std::string_view kGoVideoSessionPrefix = "go-video-";

constexpr uint32_t kProcessedVideoBitrate = 2'500'000;
constexpr const char* kProcessedVideoTrackLabel = "processed-video";
constexpr const char* kStatsChannelLabel = "cpp-processor-stats";
constexpr const char* kControlChannelLabel = "control";
constexpr const char* kDetectionChannelLabel = "detections";

}  // namespace jrb::adapters::webrtc

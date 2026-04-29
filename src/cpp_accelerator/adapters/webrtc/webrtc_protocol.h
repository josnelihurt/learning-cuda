#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <rtc/rtc.hpp>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"

namespace jrb::application::engine {
class ProcessorEngine;
}

namespace jrb::adapters::webrtc::internal {

// ---- Constants ----
constexpr std::string_view kGoVideoSessionPrefix = "go-video-";
constexpr uint32_t kProcessedVideoBitrate = 2'500'000;
constexpr const char* kProcessedVideoTrackLabel = "processed-video";
constexpr const char* kStatsChannelLabel = "cpp-processor-stats";
constexpr const char* kControlChannelLabel = "control";

// ---- Protocol conversion ----
cuda_learning::GenericFilterParameterType ConvertParamType(const std::string& type);

void CopyValidationRulesFromMetadata(const cuda_learning::FilterParameter& source,
                                     cuda_learning::GenericFilterParameter* target);

bool ParseDataChannelRequest(const std::vector<std::byte>& assembled,
                             cuda_learning::ProcessImageRequest* process_request,
                             bool* is_keepalive);

void CopyProcessMetadata(const cuda_learning::ProcessImageRequest& request,
                         cuda_learning::ProcessImageResponse* response);

// ---- Response population ----
void PopulateGetVersionResponse(jrb::application::engine::ProcessorEngine* engine,
                                cuda_learning::GetVersionInfoResponse* response);

void PopulateListFiltersResponse(jrb::application::engine::ProcessorEngine* engine,
                                 const cuda_learning::ListFiltersRequest& req,
                                 cuda_learning::ListFiltersResponse* resp);

// ---- Session utilities ----
bool IsGoVideoSession(const std::string& value);
bool ShouldRegisterSessionChannel(const std::string& session_id, const std::string& label);
std::string AcceleratorTypeLabel(cuda_learning::AcceleratorType type);

// ---- Filter parameter parsing ----
std::string NormalizeFilterId(const std::string& value);

std::optional<std::string> FirstGenericValue(
    const cuda_learning::GenericFilterParameterSelection& selection);

cuda_learning::GrayscaleType MapStringToGrayscaleType(const std::string& value);
cuda_learning::BorderMode MapStringToBorderMode(const std::string& value);
bool ParseBool(const std::string& value);
void ResolveGenericSelectionsInPlace(cuda_learning::ProcessImageRequest* request);

// ---- Transport ----
void SendFramed(rtc::DataChannel& dc, const std::string& payload, uint32_t message_id);
int64_t CurrentUnixTimeMs();

// ---- SDP / codec utilities ----
std::string NormalizeCodecName(const std::string& value);
std::string StripRtpHeaderExtensions(const std::string& sdp);
uint32_t MakeSsrc(const std::string& session_id);

// Reads WEBRTC_PUBLIC_IP / WEBRTC_PUBLIC_PORT / WEBRTC_PUBLIC_TCP_PORT env vars and builds
// the "a=candidate:..." lines to inject into the SDP answer, or returns empty string.
std::string BuildManualCandidateSdp(const std::string& session_id);

struct OutboundVideoConfig {
  std::string mid;
  int payload_type;
};

std::optional<OutboundVideoConfig> FindOutboundVideoConfig(const rtc::Description& offer);

// Waits up to 10 s for the SDP answer to be fulfilled via the promise/future or by
// polling pc.localDescription(). Returns true if sdp_answer_str was populated.
bool WaitForSdpAnswer(const std::string& session_id,
                      std::shared_future<std::string> answer_future,
                      rtc::PeerConnection& pc,
                      std::string* sdp_answer_str);

}  // namespace jrb::adapters::webrtc::internal

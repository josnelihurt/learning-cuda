#pragma once

#include <cstdint>
#include <future>
#include <optional>
#include <string>

#include <rtc/rtc.hpp>

namespace jrb::adapters::webrtc::sdp {

struct OutboundVideoConfig {
  std::string mid;
  int payload_type;
};

std::string NormalizeCodecName(const std::string& value);
std::string StripRtpHeaderExtensions(const std::string& sdp);
uint32_t MakeSsrc(const std::string& session_id);
std::optional<OutboundVideoConfig> FindOutboundVideoConfig(const rtc::Description& offer);

// Reads WEBRTC_PUBLIC_IP / WEBRTC_PUBLIC_PORT / WEBRTC_PUBLIC_TCP_PORT env vars and builds
// "a=candidate:..." lines to inject into the SDP answer, or returns empty string.
std::string BuildManualCandidateSdp(const std::string& session_id);

// Waits up to 10s for the SDP answer via future or pc.localDescription(). Returns true on success.
bool WaitForSdpAnswer(const std::string& session_id,
                      std::shared_future<std::string> answer_future,
                      rtc::PeerConnection& pc,
                      std::string* sdp_answer_str);

}  // namespace jrb::adapters::webrtc::sdp

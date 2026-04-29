#include "src/cpp_accelerator/adapters/webrtc/protocol/message_codec.h"

#include <chrono>
#include <span>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "src/cpp_accelerator/adapters/webrtc/data_channel_framing.h"

namespace jrb::adapters::webrtc::protocol {

bool ParseDataChannelRequest(const std::vector<std::byte>& assembled,
                             cuda_learning::ProcessImageRequest* process_request,
                             bool* is_keepalive) {
  if (process_request == nullptr || is_keepalive == nullptr) return false;
  *is_keepalive = false;

  cuda_learning::DataChannelRequest envelope;
  if (envelope.ParseFromArray(assembled.data(), static_cast<int>(assembled.size()))) {
    if (envelope.has_keepalive()) {
      *is_keepalive = true;
      return true;
    }
    if (envelope.has_process_image()) {
      *process_request = envelope.process_image();
      return true;
    }
    return false;
  }

  // Compatibility path: accept legacy raw ProcessImageRequest payloads.
  if (process_request->ParseFromArray(assembled.data(), static_cast<int>(assembled.size()))) {
    spdlog::warn("[WebRTC] Received legacy raw ProcessImageRequest payload; migrate client to DataChannelRequest envelope");
    return true;
  }

  return false;
}

void CopyProcessMetadata(const cuda_learning::ProcessImageRequest& request,
                         cuda_learning::ProcessImageResponse* response) {
  if (response == nullptr) return;
  response->set_api_version(request.api_version());
  response->mutable_trace_context()->CopyFrom(request.trace_context());
}

void SendFramed(rtc::DataChannel& dc, const std::string& payload, uint32_t message_id) {
  const auto span = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(payload.data()), payload.size());
  auto chunks = PackMessage(message_id, span);
  for (auto& chunk : chunks) {
    if (!dc.send(std::move(chunk))) {
      spdlog::error("[framing] Failed to send chunk for message_id={}", message_id);
      return;
    }
  }
}

int64_t CurrentUnixTimeMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace jrb::adapters::webrtc::protocol

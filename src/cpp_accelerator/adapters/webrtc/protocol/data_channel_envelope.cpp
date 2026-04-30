#include "src/cpp_accelerator/adapters/webrtc/protocol/data_channel_envelope.h"

#include <spdlog/spdlog.h>

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

}  // namespace jrb::adapters::webrtc::protocol

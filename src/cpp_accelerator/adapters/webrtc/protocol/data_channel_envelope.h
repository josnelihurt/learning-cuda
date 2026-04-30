#pragma once

#include <vector>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"

namespace jrb::adapters::webrtc::protocol {

// Decodes a fully reassembled DataChannelRequest envelope.
// Returns false if the bytes are neither a DataChannelRequest nor a legacy raw
// ProcessImageRequest. On success, sets *is_keepalive iff the envelope was a
// keepalive (process_request stays untouched), or fills *process_request with
// the inner ProcessImageRequest.
bool ParseDataChannelRequest(const std::vector<std::byte>& assembled,
                             cuda_learning::ProcessImageRequest* process_request,
                             bool* is_keepalive);

// Carries api_version + trace_context from a request to its response.
void CopyProcessMetadata(const cuda_learning::ProcessImageRequest& request,
                         cuda_learning::ProcessImageResponse* response);

}  // namespace jrb::adapters::webrtc::protocol

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rtc/rtc.hpp>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"

namespace jrb::adapters::webrtc::protocol {

bool ParseDataChannelRequest(const std::vector<std::byte>& assembled,
                             cuda_learning::ProcessImageRequest* process_request,
                             bool* is_keepalive);

void CopyProcessMetadata(const cuda_learning::ProcessImageRequest& request,
                         cuda_learning::ProcessImageResponse* response);

void SendFramed(rtc::DataChannel& dc, const std::string& payload, uint32_t message_id);

int64_t CurrentUnixTimeMs();

}  // namespace jrb::adapters::webrtc::protocol

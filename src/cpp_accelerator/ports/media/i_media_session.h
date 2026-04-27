#pragma once

#include <string>

namespace jrb::ports::media {

// Driven port — manages WebRTC media sessions on behalf of the control client.
// Concrete implementations live in adapters/webrtc/.
class IMediaSession {
 public:
  virtual ~IMediaSession() = default;

  virtual bool Initialize() = 0;
  virtual void Shutdown() = 0;

  virtual bool CreateSession(const std::string& session_id, const std::string& sdp_offer,
                             std::string* sdp_answer, std::string* error_message) = 0;

  virtual bool CloseSession(const std::string& session_id, std::string* error_message) = 0;

  virtual bool HandleRemoteCandidate(const std::string& session_id, const std::string& candidate,
                                     const std::string& sdp_mid, int sdp_mline_index,
                                     std::string* error_message) = 0;
};

}  // namespace jrb::ports::media

#include "src/cpp_accelerator/adapters/webrtc/session_routing.h"

#include "src/cpp_accelerator/adapters/webrtc/channel_labels.h"

namespace jrb::adapters::webrtc {

bool IsGoVideoSession(const std::string& value) {
  return value.rfind(kGoVideoSessionPrefix, 0) == 0;
}

bool ShouldRegisterSessionChannel(const std::string& session_id, const std::string& label) {
  return !IsGoVideoSession(session_id) && !IsGoVideoSession(label);
}

}  // namespace jrb::adapters::webrtc

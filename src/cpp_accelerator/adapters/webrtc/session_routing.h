#pragma once

#include <string>

namespace jrb::adapters::webrtc {

bool IsGoVideoSession(const std::string& value);

// True if the data channel should be registered as the routable response
// channel for `session_id` — i.e. neither the session ID nor the channel
// label looks like a Go-originated video relay.
bool ShouldRegisterSessionChannel(const std::string& session_id, const std::string& label);

}  // namespace jrb::adapters::webrtc

#include "cpp_accelerator/ports/grpc/webrtc_manager.h"

#include <spdlog/spdlog.h>
#include <chrono>
#include <future>
#include <rtc/rtc.hpp>
#include <thread>

namespace jrb::ports::grpc_service {

WebRTCManager::WebRTCManager() : initialized_(false), cleanup_running_(false) {
  rtc::InitLogger(rtc::LogLevel::Info);
}

WebRTCManager::~WebRTCManager() {
  if (initialized_) {
    Shutdown();
  }
}

bool WebRTCManager::Initialize() {
  if (initialized_) {
    spdlog::warn("WebRTCManager already initialized");
    return true;
  }

  try {
    spdlog::info("Initializing WebRTCManager with libdatachannel");
    config_ = std::make_unique<rtc::Configuration>();
    config_->iceServers.emplace_back("stun:stun.l.google.com:19302");

    config_->portRangeBegin = 10000;
    config_->portRangeEnd = 10199;

    spdlog::info("WebRTC Configuration:");
    spdlog::info("  - STUN Server: stun.l.google.com:19302");
    spdlog::info("  - UDP Port Range: {}-{} ({} ports)", config_->portRangeBegin,
                 config_->portRangeEnd, config_->portRangeEnd - config_->portRangeBegin + 1);
    spdlog::info("  - TURN Server: Not configured");
    spdlog::info("  - Session Cleanup: Enabled (30s timeout)");

    cleanup_running_ = true;
    cleanup_thread_ = std::thread([this]() {
      while (cleanup_running_) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (cleanup_running_) {
          CleanupInactiveSessions(30);
        }
      }
    });

    initialized_ = true;
    spdlog::info("WebRTCManager initialized successfully");
    return true;
  } catch (const std::exception& e) {
    spdlog::error("Failed to initialize WebRTCManager: {}", e.what());
    initialized_ = false;
    return false;
  }
}

void WebRTCManager::Shutdown() {
  if (!initialized_) {
    return;
  }

  spdlog::info("Shutting down WebRTCManager...");

  cleanup_running_ = false;
  if (cleanup_thread_.joinable()) {
    cleanup_thread_.join();
  }

  std::lock_guard<std::mutex> lock(sessions_mutex_);
  for (auto& [session_id, session] : sessions_) {
    try {
      std::lock_guard<std::mutex> session_lock(session->mutex);
      if (session->data_channel) {
        session->data_channel->close();
      }
      if (session->peer_connection) {
        session->peer_connection->close();
      }
    } catch (const std::exception& e) {
      spdlog::error("[WebRTC:{}] Error closing session during shutdown: {}", session_id, e.what());
    }
  }
  sessions_.clear();
  config_.reset();
  initialized_ = false;
  spdlog::info("WebRTCManager shut down");
}

bool WebRTCManager::CreateSession(const std::string& session_id, const std::string& sdp_offer_str,
                                  std::string* sdp_answer_str, std::string* error_message) {
  if (!initialized_) {
    if (error_message != nullptr) {
      *error_message = "WebRTC manager not initialized";
    }
    return false;
  }

  if (session_id.empty() || sdp_offer_str.empty()) {
    if (error_message != nullptr) {
      *error_message = "session_id and sdp_offer are required";
    }
    return false;
  }

  std::lock_guard<std::mutex> lock(sessions_mutex_);
  if (sessions_.count(session_id)) {
    if (error_message != nullptr) {
      *error_message = "Session with ID " + session_id + " already exists";
    }
    return false;
  }

  try {
    auto session = std::make_shared<SessionState>();
    session->created_at = std::chrono::steady_clock::now();
    session->last_heartbeat = std::chrono::steady_clock::now();
    session->peer_connection = std::make_shared<rtc::PeerConnection>(*config_);

    session->peer_connection->onStateChange(
        [session_id, session](rtc::PeerConnection::State state) {
          std::string state_str;
          switch (state) {
            case rtc::PeerConnection::State::New:
              state_str = "New";
              break;
            case rtc::PeerConnection::State::Connecting:
              state_str = "Connecting";
              break;
            case rtc::PeerConnection::State::Connected:
              state_str = "Connected";
              break;
            case rtc::PeerConnection::State::Disconnected:
              state_str = "Disconnected";
              spdlog::warn("[WebRTC:{}] Peer connection disconnected", session_id);
              break;
            case rtc::PeerConnection::State::Failed:
              state_str = "Failed";
              spdlog::error("[WebRTC:{}] Peer connection failed", session_id);
              break;
            case rtc::PeerConnection::State::Closed:
              state_str = "Closed";
              spdlog::info("[WebRTC:{}] Peer connection closed", session_id);
              break;
            default:
              state_str = "Unknown";
              break;
          }
          spdlog::info("[WebRTC:{}] Peer connection state changed: {} ({})", session_id, state_str,
                       static_cast<int>(state));
        });

    session->peer_connection->onGatheringStateChange(
        [session_id](rtc::PeerConnection::GatheringState state) {
          std::string state_str;
          switch (state) {
            case rtc::PeerConnection::GatheringState::New:
              state_str = "New";
              break;
            case rtc::PeerConnection::GatheringState::InProgress:
              state_str = "InProgress";
              break;
            case rtc::PeerConnection::GatheringState::Complete:
              state_str = "Complete";
              break;
            default:
              state_str = "Unknown";
              break;
          }
          spdlog::info("[WebRTC:{}] ICE gathering state changed: {} ({})", session_id, state_str,
                       static_cast<int>(state));
        });

    session->peer_connection->onIceStateChange(
        [session_id, session](rtc::PeerConnection::IceState state) {
          std::string state_str;
          switch (state) {
            case rtc::PeerConnection::IceState::New:
              state_str = "New";
              break;
            case rtc::PeerConnection::IceState::Checking:
              state_str = "Checking";
              break;
            case rtc::PeerConnection::IceState::Connected:
              state_str = "Connected";
              spdlog::info("[WebRTC:{}] ICE connection established", session_id);
              break;
            case rtc::PeerConnection::IceState::Completed:
              state_str = "Completed";
              spdlog::info("[WebRTC:{}] ICE connection completed", session_id);
              break;
            case rtc::PeerConnection::IceState::Failed:
              state_str = "Failed";
              spdlog::error("[WebRTC:{}] ICE connection failed", session_id);
              break;
            case rtc::PeerConnection::IceState::Disconnected:
              state_str = "Disconnected";
              spdlog::warn("[WebRTC:{}] ICE connection disconnected", session_id);
              break;
            case rtc::PeerConnection::IceState::Closed:
              state_str = "Closed";
              spdlog::info("[WebRTC:{}] ICE connection closed", session_id);
              break;
            default:
              state_str = "Unknown";
              break;
          }
          spdlog::info("[WebRTC:{}] ICE state changed: {} ({})", session_id, state_str,
                       static_cast<int>(state));
        });

    // Store answer pointer for async callback
    std::string* answer_ptr = sdp_answer_str;
    auto answer_promise = std::make_shared<std::promise<std::string>>();
    std::shared_future<std::string> answer_future = answer_promise->get_future();

    session->peer_connection->onLocalDescription(
        [session_id, answer_ptr, answer_promise](rtc::Description description) {
          spdlog::info("[WebRTC:{}] Local description created (type: {})", session_id,
                       description.typeString());
          if (description.type() == rtc::Description::Type::Answer) {
            std::string sdp = description.generateSdp();
            if (answer_ptr != nullptr && answer_ptr->empty()) {
              *answer_ptr = sdp;
            }
            try {
              answer_promise->set_value(sdp);
              spdlog::info("[WebRTC:{}] SDP answer generated and stored (length: {})", session_id,
                           sdp.length());
            } catch (const std::future_error& e) {
              spdlog::warn("[WebRTC:{}] Promise already set: {}", session_id, e.what());
            }
          } else {
            spdlog::warn("[WebRTC:{}] Local description is not Answer type: {}", session_id,
                         description.typeString());
          }
        });

    session->peer_connection->onLocalCandidate([session_id](rtc::Candidate candidate) {
      spdlog::info("[WebRTC:{}] Local ICE candidate: {}", session_id, candidate.candidate());
    });

    // Register callback to receive remote data channel from client
    // This must be done BEFORE setRemoteDescription to ensure we capture the channel
    session->peer_connection->onDataChannel([session_id, session](
                                                std::shared_ptr<rtc::DataChannel> data_channel) {
      spdlog::info("[WebRTC:{}] Remote data channel received: {}", session_id,
                   data_channel->label());
      session->data_channel = data_channel;

      // Register callbacks on the received data channel
      data_channel->onOpen([session_id, session]() {
        spdlog::info("[WebRTC:{}] Remote data channel opened successfully - isOpen: {}", session_id,
                     session->data_channel ? session->data_channel->isOpen() : false);
      });

      data_channel->onMessage([session_id, session](rtc::message_variant data) {
        spdlog::info("[WebRTC:{}] onMessage callback triggered", session_id);
        if (std::holds_alternative<std::string>(data)) {
          std::string message = std::get<std::string>(data);
          spdlog::info("[WebRTC:{}] Received message on data channel: '{}' (length: {})",
                       session_id, message, message.length());
          if (message == "ping") {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->data_channel && session->data_channel->isOpen()) {
              session->last_heartbeat = std::chrono::steady_clock::now();
              bool send_result = session->data_channel->send("pong");
              spdlog::info("[WebRTC:{}] Sent pong response (result: {}) and updated heartbeat",
                           session_id, send_result);
            } else {
              spdlog::warn("[WebRTC:{}] Cannot send pong - data channel not open (isOpen: {})",
                           session_id,
                           session->data_channel ? session->data_channel->isOpen() : false);
            }
          } else {
            spdlog::warn("[WebRTC:{}] Received non-ping message: '{}'", session_id, message);
          }
        } else {
          spdlog::warn("[WebRTC:{}] Received non-string message (variant type)", session_id);
        }
      });

      data_channel->onClosed(
          [session_id]() { spdlog::warn("[WebRTC:{}] Remote data channel closed", session_id); });

      data_channel->onError([session_id](std::string err) {
        spdlog::error("[WebRTC:{}] Remote data channel error: {}", session_id, err);
      });

      spdlog::info("[WebRTC:{}] Data channel callbacks registered on remote channel", session_id);
    });

    // Set remote description - this will trigger onDataChannel if client created a channel
    spdlog::debug("[WebRTC:{}] Parsing SDP offer (length: {})", session_id, sdp_offer_str.length());

    // Check if SDP contains fingerprint (required for DTLS)
    bool has_fingerprint = sdp_offer_str.find("fingerprint:") != std::string::npos ||
                           sdp_offer_str.find("a=fingerprint:") != std::string::npos;
    spdlog::debug("[WebRTC:{}] SDP offer contains fingerprint: {}", session_id, has_fingerprint);

    if (!has_fingerprint) {
      spdlog::warn(
          "[WebRTC:{}] SDP offer missing fingerprint - this may cause DTLS handshake to fail",
          session_id);
      spdlog::debug("[WebRTC:{}] SDP offer preview: {}", session_id, sdp_offer_str.substr(0, 300));
    }

    rtc::Description offer(sdp_offer_str, rtc::Description::Type::Offer);
    spdlog::debug("[WebRTC:{}] SDP offer parsed successfully, type: {}", session_id,
                  offer.typeString());

    try {
      session->peer_connection->setRemoteDescription(offer);
      spdlog::info("[WebRTC:{}] Remote description set successfully", session_id);
    } catch (const std::exception& e) {
      spdlog::error("[WebRTC:{}] Failed to set remote description: {}", session_id, e.what());
      spdlog::error("[WebRTC:{}] Full SDP offer that failed:\n{}", session_id, sdp_offer_str);
      if (error_message != nullptr) {
        *error_message = std::string("Failed to set remote description: ") + e.what();
      }
      return false;
    }

    // Wait a moment for the state to transition and for onDataChannel to be called
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    for (const auto& candidate : session->pending_candidates) {
      session->peer_connection->addRemoteCandidate(candidate);
    }
    session->pending_candidates.clear();

    // Create answer - libdatachannel generates it asynchronously
    // The signaling state should now be "have-remote-offer" which allows creating an answer
    // Note: setLocalDescription may throw an exception, but the callback will still fire
    try {
      session->peer_connection->setLocalDescription(rtc::Description::Type::Answer);
    } catch (const std::exception& e) {
      spdlog::warn("[WebRTC:{}] setLocalDescription threw exception (may be non-fatal): {}",
                   session_id, e.what());
      // Continue anyway - the callback may still fire and generate the answer
      // libdatachannel sometimes throws but still generates the answer
    }

    // Check immediately if answer is available (libdatachannel may generate it synchronously)
    // Then wait for callback with timeout
    auto start_time = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(10);
    bool answer_ready = false;

    // First, check immediately
    if (session->peer_connection->localDescription().has_value()) {
      auto local_desc = session->peer_connection->localDescription().value();
      if (local_desc.type() == rtc::Description::Type::Answer) {
        std::string sdp = local_desc.generateSdp();
        if (sdp_answer_str != nullptr && sdp_answer_str->empty()) {
          *sdp_answer_str = sdp;
          spdlog::info("[WebRTC:{}] SDP answer available immediately (length: {})", session_id,
                       sdp.length());
          answer_ready = true;
        }
      }
    }

    // If not available immediately, wait for callback with periodic checks
    if (!answer_ready) {
      spdlog::info("[WebRTC:{}] Waiting for SDP answer (timeout: {}s)", session_id,
                   timeout.count());
      while ((std::chrono::steady_clock::now() - start_time) < timeout) {
        // Check if callback fired
        auto status = answer_future.wait_for(std::chrono::milliseconds(100));
        if (status == std::future_status::ready) {
          try {
            std::string answer = answer_future.get();
            if (sdp_answer_str != nullptr && sdp_answer_str->empty()) {
              *sdp_answer_str = answer;
            }
            spdlog::info("[WebRTC:{}] SDP answer received via callback (length: {})", session_id,
                         answer.length());
            answer_ready = true;
            break;
          } catch (const std::exception& e) {
            spdlog::error("[WebRTC:{}] Error getting answer from future: {}", session_id, e.what());
          }
        }

        // Also check directly (callback may have fired but promise not set yet)
        if (session->peer_connection->localDescription().has_value()) {
          auto local_desc = session->peer_connection->localDescription().value();
          if (local_desc.type() == rtc::Description::Type::Answer) {
            std::string sdp = local_desc.generateSdp();
            if (sdp_answer_str != nullptr && sdp_answer_str->empty()) {
              *sdp_answer_str = sdp;
              spdlog::info("[WebRTC:{}] Retrieved SDP answer directly (length: {})", session_id,
                           sdp.length());
              answer_ready = true;
              break;
            }
          }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }

    // If we got the answer, consider it successful even if setLocalDescription threw
    if (answer_ready && sdp_answer_str != nullptr && !sdp_answer_str->empty()) {
      spdlog::info("[WebRTC:{}] Session created successfully (answer length: {})", session_id,
                   sdp_answer_str->length());
      sessions_[session_id] = session;
      return true;
    }

    if (!answer_ready) {
      spdlog::error("[WebRTC:{}] Timeout waiting for SDP answer ({}s)", session_id,
                    timeout.count());
      // Clean up the session before returning
      try {
        std::lock_guard<std::mutex> session_lock(session->mutex);
        if (session->data_channel) {
          session->data_channel->close();
        }
        if (session->peer_connection) {
          session->peer_connection->close();
        }
      } catch (const std::exception& e) {
        spdlog::warn("[WebRTC:{}] Error cleaning up failed session: {}", session_id, e.what());
      }
      RemoveSession(session_id);
      if (error_message != nullptr) {
        *error_message =
            "Timeout waiting for SDP answer after " + std::to_string(timeout.count()) + " seconds";
      }
      return false;
    }

    // This should not be reached if answer_ready is true (we return early above)
    // But keep it as fallback
    sessions_[session_id] = session;
    spdlog::info("[WebRTC:{}] Session created successfully", session_id);
    return true;
  } catch (const std::exception& e) {
    if (error_message != nullptr) {
      *error_message = "Failed to create WebRTC session: " + std::string(e.what());
    }
    spdlog::error("[WebRTC:{}] Failed to create session: {}", session_id, e.what());
    RemoveSession(session_id);
    return false;
  }
}

bool WebRTCManager::CloseSession(const std::string& session_id, std::string* error_message) {
  auto session = GetSession(session_id);
  if (!session) {
    if (error_message != nullptr) {
      *error_message = "Session with ID " + session_id + " not found";
    }
    return false;
  }

  try {
    std::lock_guard<std::mutex> lock(session->mutex);

    if (session->data_channel) {
      session->data_channel->close();
      spdlog::info("[WebRTC:{}] Data channel closed", session_id);
    }

    if (session->peer_connection) {
      session->peer_connection->close();
      spdlog::info("[WebRTC:{}] Peer connection closed", session_id);
    }

    RemoveSession(session_id);
    spdlog::info("[WebRTC:{}] Session closed successfully", session_id);
    return true;
  } catch (const std::exception& e) {
    if (error_message != nullptr) {
      *error_message = "Failed to close session: " + std::string(e.what());
    }
    spdlog::error("[WebRTC:{}] Failed to close session: {}", session_id, e.what());
    RemoveSession(session_id);
    return false;
  }
}

bool WebRTCManager::HandleRemoteCandidate(const std::string& session_id,
                                          const std::string& candidate_str,
                                          const std::string& sdp_mid, int /* sdp_mline_index */,
                                          std::string* error_message) {
  auto session = GetSession(session_id);
  if (!session) {
    if (error_message != nullptr) {
      *error_message = "Session with ID " + session_id + " not found";
    }
    return false;
  }

  try {
    rtc::Candidate candidate(candidate_str, sdp_mid);
    std::lock_guard<std::mutex> lock(session->mutex);

    if (session->peer_connection->remoteDescription().has_value()) {
      session->peer_connection->addRemoteCandidate(candidate);
      spdlog::info("[WebRTC:{}] Added remote ICE candidate: {}", session_id, candidate_str);
    } else {
      session->pending_candidates.push_back(candidate);
      spdlog::info("[WebRTC:{}] Stored pending remote ICE candidate: {}", session_id,
                   candidate_str);
    }
    return true;
  } catch (const std::exception& e) {
    if (error_message != nullptr) {
      *error_message = "Failed to add remote ICE candidate: " + std::string(e.what());
    }
    spdlog::error("[WebRTC:{}] Failed to add candidate: {}", session_id, e.what());
    return false;
  }
}

std::shared_ptr<WebRTCManager::SessionState> WebRTCManager::GetSession(
    const std::string& session_id) {
  std::lock_guard<std::mutex> lock(sessions_mutex_);
  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    return it->second;
  }
  return nullptr;
}

void WebRTCManager::RemoveSession(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(sessions_mutex_);
  sessions_.erase(session_id);
  spdlog::info("[WebRTC:{}] Session removed", session_id);
}

void WebRTCManager::CleanupInactiveSessions(int timeout_seconds) {
  std::lock_guard<std::mutex> lock(sessions_mutex_);
  auto now = std::chrono::steady_clock::now();
  auto timeout_duration = std::chrono::seconds(timeout_seconds);

  std::vector<std::string> sessions_to_remove;

  for (const auto& [session_id, session] : sessions_) {
    std::lock_guard<std::mutex> session_lock(session->mutex);
    auto elapsed = now - session->last_heartbeat;

    if (elapsed > timeout_duration) {
      spdlog::warn("[WebRTC:{}] Session inactive for {} seconds, closing", session_id,
                   std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
      sessions_to_remove.push_back(session_id);

      try {
        if (session->data_channel) {
          session->data_channel->close();
        }
        if (session->peer_connection) {
          session->peer_connection->close();
        }
      } catch (const std::exception& e) {
        spdlog::error("[WebRTC:{}] Error closing inactive session: {}", session_id, e.what());
      }
    }
  }

  for (const auto& session_id : sessions_to_remove) {
    sessions_.erase(session_id);
    spdlog::info("[WebRTC:{}] Inactive session removed", session_id);
  }

  if (!sessions_to_remove.empty()) {
    spdlog::info("Cleaned up {} inactive WebRTC session(s)", sessions_to_remove.size());
  }
}

}  // namespace jrb::ports::grpc_service

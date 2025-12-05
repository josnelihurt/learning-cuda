#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rtc/rtc.hpp>

namespace jrb::ports::grpc_service {

class WebRTCManager {
public:
  WebRTCManager();
  ~WebRTCManager();

  bool Initialize();
  void Shutdown();

  bool IsInitialized() const { return initialized_; }

  bool CreateSession(const std::string& session_id, const std::string& sdp_offer,
                     std::string* sdp_answer, std::string* error_message);

  bool CloseSession(const std::string& session_id, std::string* error_message);

  bool HandleRemoteCandidate(const std::string& session_id, const std::string& candidate,
                             const std::string& sdp_mid, int sdp_mline_index,
                             std::string* error_message);

private:
  struct SessionState {
    std::shared_ptr<rtc::PeerConnection> peer_connection;
    std::shared_ptr<rtc::DataChannel> data_channel;
    std::vector<rtc::Candidate> pending_candidates;
    std::mutex mutex;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_heartbeat;
  };

  std::shared_ptr<SessionState> GetSession(const std::string& session_id);
  void RemoveSession(const std::string& session_id);
  void CleanupInactiveSessions(int timeout_seconds = 30);

  bool initialized_;
  std::unique_ptr<rtc::Configuration> config_;
  std::mutex sessions_mutex_;
  std::unordered_map<std::string, std::shared_ptr<SessionState>> sessions_;
  std::atomic<bool> cleanup_running_;
  std::thread cleanup_thread_;
};

}  // namespace jrb::ports::grpc_service

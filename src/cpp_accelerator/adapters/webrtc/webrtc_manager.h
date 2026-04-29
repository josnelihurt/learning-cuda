#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rtc/rtc.hpp>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#include "src/cpp_accelerator/adapters/compute/cuda/memory/cuda_memory_pool.h"
#include "src/cpp_accelerator/adapters/webrtc/data_channel_framing.h"
#include "src/cpp_accelerator/adapters/webrtc/live_video_processor.h"
#include "src/cpp_accelerator/application/server_info/i_server_info_provider.h"
#include "src/cpp_accelerator/ports/media/i_media_session.h"

namespace jrb::application::engine {
class ProcessorEngine;
}

namespace jrb::adapters::webrtc {

struct WebRTCManagerConfig {
  std::shared_ptr<jrb::application::engine::ProcessorEngine> engine;
  std::string device_id;
  std::string display_name;
};

class WebRTCManager : public std::enable_shared_from_this<WebRTCManager>,
                      public jrb::ports::media::IMediaSession {
public:
  explicit WebRTCManager(WebRTCManagerConfig config = {});
  virtual ~WebRTCManager();

  bool Initialize();
  void Shutdown();

  bool IsInitialized() const { return initialized_; }

  bool CreateSession(const std::string& session_id, const std::string& sdp_offer,
                     std::string* sdp_answer, std::string* error_message);

  bool CloseSession(const std::string& session_id, std::string* error_message);

  bool HandleRemoteCandidate(const std::string& session_id, const std::string& candidate,
                             const std::string& sdp_mid, int sdp_mline_index,
                             std::string* error_message);

  std::vector<rtc::Candidate> GetPendingLocalCandidates(const std::string& session_id);
  virtual void SendToSession(const std::string& session_id, const std::string& bytes);

  private:
  struct SessionState {
    std::shared_ptr<rtc::PeerConnection> peer_connection;
    std::shared_ptr<rtc::DataChannel> data_channel;
    std::shared_ptr<rtc::DataChannel> detection_channel;
    std::shared_ptr<rtc::DataChannel> stats_channel;
    std::shared_ptr<rtc::DataChannel> control_channel;
    std::shared_ptr<rtc::Track> inbound_video_track;
    std::shared_ptr<rtc::Track> outbound_video_track;
    std::shared_ptr<rtc::RtcpReceivingSession> inbound_rtcp_session;
    std::shared_ptr<rtc::H264RtpDepacketizer> inbound_depacketizer;
    std::shared_ptr<rtc::RtpPacketizationConfig> outbound_rtp_config;
    std::shared_ptr<rtc::H264RtpPacketizer> outbound_packetizer;
    std::shared_ptr<rtc::RtcpSrReporter> outbound_sr_reporter;
    std::shared_ptr<rtc::RtcpNackResponder> outbound_nack_responder;
    // Lifetime invariant: memory_pool MUST be declared before
    // live_video_processor so the pool outlives the processor (members are
    // destroyed in reverse declaration order, and the processor holds a raw
    // pointer into the pool).
    std::unique_ptr<jrb::infrastructure::cuda::CudaMemoryPool> memory_pool;
    std::unique_ptr<LiveVideoProcessor> live_video_processor;
    std::unique_ptr<ChunkReassembler> incoming_reassembler =
        std::make_unique<ChunkReassembler>();
    std::atomic<uint32_t> outgoing_message_id{0};
    std::atomic<int> frame_count{0};
    cuda_learning::ProcessImageRequest live_filter_state;
    std::vector<rtc::Candidate> pending_candidates;
    std::queue<rtc::Candidate> local_candidates_queue;
    std::mutex mutex;
    std::mutex media_mutex;
    std::mutex candidates_mutex;
    std::condition_variable candidates_cv;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_heartbeat;
  };

  std::shared_ptr<SessionState> GetSession(const std::string& session_id);
  void RemoveSession(const std::string& session_id);
  void CleanupInactiveSessions(int timeout_seconds = 30);

  // Close every data channel and the peer connection on `session`. Idempotent
  // and exception-safe — single point of truth used by Shutdown(),
  // CloseSession(), and CleanupInactiveSessions(). Caller is responsible for
  // any necessary locking.
  static void CloseSessionTransport(const std::string& session_id, SessionState& session);
  void RegisterSessionChannel(const std::string& session_id,
                              const std::shared_ptr<rtc::DataChannel>& data_channel);
  void UnregisterSessionChannel(const std::string& session_id);

  // Returns a shared_future that will be fulfilled with the SDP answer string.
  // Registers all peer connection state callbacks (onStateChange, onGatheringStateChange,
  // onIceStateChange, onLocalDescription, onLocalCandidate) and the onTrack callback.
  std::shared_future<std::string> SetupPeerConnectionCallbacks(
      const std::string& session_id,
      std::shared_ptr<SessionState> session,
      std::shared_ptr<std::string> manual_candidate_sdp,
      std::string* sdp_answer_out);

  // Adds the outbound H264 video track and configures the RTP packetizer chain
  // based on the parsed SDP offer. Returns false on error.
  bool SetupMediaTracks(const std::string& session_id,
                        std::shared_ptr<SessionState> session,
                        const rtc::Description& offer,
                        std::string* error_message);

  // Registers the onDataChannel callback that routes the four data channels
  // (main, detections, stats, control) to their respective handlers.
  void SetupDataChannels(const std::string& session_id, std::shared_ptr<SessionState> session);

  // Handles a framed binary payload from the control channel.
  void HandleControlMessage(const std::string& session_id,
                            SessionState& state,
                            const rtc::binary& raw_chunk,
                            rtc::DataChannel& response_channel);

  // Handles a framed binary payload from the main data channel (image processing requests).
  void HandleProcessingMessage(const std::string& session_id,
                               SessionState& state,
                               const rtc::binary& raw_chunk,
                               rtc::DataChannel& response_channel);

  // Processes a decoded inbound video frame and sends the result on the outbound track.
  void HandleVideoFrame(const std::string& session_id,
                        SessionState& state,
                        rtc::binary frame,
                        rtc::FrameInfo info);

  // Sets up RTP handlers and callbacks for an inbound RecvOnly video track.
  void SetupInboundTrackCallbacks(const std::string& session_id,
                                  std::shared_ptr<SessionState> session,
                                  std::shared_ptr<rtc::Track> track);

  // Per-channel setup helpers called from SetupDataChannels.
  void SetupDetectionChannel(const std::string& session_id,
                             std::shared_ptr<SessionState> session,
                             std::shared_ptr<rtc::DataChannel> dc);
  void SetupControlChannel(const std::string& session_id,
                           std::shared_ptr<SessionState> session,
                           std::shared_ptr<rtc::DataChannel> dc);
  void SetupStatsChannel(const std::string& session_id,
                         std::shared_ptr<SessionState> session,
                         std::shared_ptr<rtc::DataChannel> dc);
  void SetupMainChannel(const std::string& session_id,
                        std::shared_ptr<SessionState> session,
                        std::shared_ptr<rtc::DataChannel> dc);

  // Serializes and sends a ProcessingStatsFrame to the stats channel if open.
  void EmitProcessingStats(const std::string& session_id, SessionState& state,
                           double elapsed_ms, int64_t detection_count, uint32_t frame_id);

  // Serializes and sends a DetectionFrame to the detection channel if open.
  void ForwardDetections(const std::string& session_id, SessionState& state,
                         const cuda_learning::DetectionFrame& frame);

  std::shared_ptr<jrb::application::engine::ProcessorEngine> engine_;
  std::unique_ptr<jrb::application::server_info::IServerInfoProvider> server_info_;
  bool initialized_;
  std::unique_ptr<rtc::Configuration> config_;
  std::mutex sessions_mutex_;
  std::unordered_map<std::string, std::shared_ptr<SessionState>> sessions_;
  std::mutex session_channels_mutex_;
  std::unordered_map<std::string, std::weak_ptr<rtc::DataChannel>> session_channels_;
  std::string device_id_;
  std::string display_name_;
  std::atomic<bool> cleanup_running_;
  std::thread cleanup_thread_;
};

}  // namespace jrb::adapters::webrtc

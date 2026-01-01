#include "cpp_accelerator/ports/grpc/webrtc_signaling_service_impl.h"

#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>

#include <spdlog/spdlog.h>

// Threading and synchronization:
// - candidate_sender: Reads from thread-safe queue and sends to stream
// - keepalive_thread: Sends keepalive every 30 seconds
// - Main loop: Receives messages from client and processes them
// - stream_mutex: Protects stream write operations from concurrent access

namespace jrb::ports::grpc_service {

WebRTCSignalingServiceImpl::WebRTCSignalingServiceImpl(std::shared_ptr<WebRTCManager> manager)
    : manager_(std::move(manager)) {}

::grpc::Status WebRTCSignalingServiceImpl::SignalingStream(
    ::grpc::ServerContext* context,
    ::grpc::ServerReaderWriter<cuda_learning::SignalingMessage,
                               cuda_learning::SignalingMessage>* stream) {
  if (manager_ == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "WebRTC manager not available");
  }

  if (!manager_->IsInitialized()) {
    return ::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION,
                          "WebRTC manager not initialized");
  }

  std::string current_session_id;
  std::mutex stream_mutex;
  std::atomic<bool> stream_active{true};
  std::unordered_map<std::string, std::string> session_to_stream_map;

  auto send_keepalive = [&]() {
    cuda_learning::SignalingMessage msg;
    auto* keepalive = msg.mutable_keep_alive();
    keepalive->set_timestamp(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());

    std::lock_guard<std::mutex> lock(stream_mutex);
    if (stream_active.load() && !context->IsCancelled()) {
      stream->Write(msg);
    }
  };

  std::thread candidate_sender([&]() {
    while (!context->IsCancelled() && stream_active.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      if (current_session_id.empty()) {
        continue;
      }

      auto candidates = manager_->GetPendingLocalCandidates(current_session_id);
      if (!candidates.empty()) {
        std::lock_guard<std::mutex> lock(stream_mutex);
        if (!stream_active.load() || context->IsCancelled()) {
          break;
        }

        for (const auto& candidate : candidates) {
          cuda_learning::SignalingMessage msg;
          auto* candidate_req = msg.mutable_ice_candidate();
          candidate_req->set_session_id(current_session_id);
          candidate_req->mutable_candidate()->set_candidate(candidate.candidate());
          candidate_req->mutable_candidate()->set_sdp_mid(candidate.mid());
          candidate_req->mutable_candidate()->set_sdp_mline_index(0);

          if (!stream->Write(msg)) {
            spdlog::warn("[WebRTC:{}] Failed to send local candidate to stream",
                         current_session_id);
            stream_active = false;
            break;
          }
        }
      }
    }
  });

  std::thread keepalive_thread([&]() {
    while (!context->IsCancelled() && stream_active.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(30));
      if (!context->IsCancelled() && stream_active.load()) {
        send_keepalive();
      }
    }
  });

  cuda_learning::SignalingMessage msg;
  while (stream->Read(&msg)) {
    if (context->IsCancelled()) {
      break;
    }

    switch (msg.message_case()) {
      case cuda_learning::SignalingMessage::kStartSession: {
        const auto& req = msg.start_session();
        current_session_id = req.session_id();
        session_to_stream_map[current_session_id] = current_session_id;

        std::string answer;
        std::string error;
        const bool success =
            manager_->CreateSession(req.session_id(), req.sdp_offer(), &answer, &error);

        cuda_learning::SignalingMessage response;
        auto* resp = response.mutable_start_session_response();
        resp->set_session_id(req.session_id());
        if (success) {
          resp->set_sdp_answer(answer);
          if (req.has_trace_context()) {
            *resp->mutable_trace_context() = req.trace_context();
          }
        } else {
          if (error.empty()) {
            error = "failed to create WebRTC session";
          }
          resp->set_sdp_answer("");
        }

        {
          std::lock_guard<std::mutex> lock(stream_mutex);
          if (!stream_active.load()) {
            break;
          }
          if (!stream->Write(response)) {
            spdlog::error("[WebRTC:{}] Failed to send start_session response",
                           req.session_id());
            stream_active = false;
            break;
          }
        }

        spdlog::info("[WebRTC:{}] Session started via stream", req.session_id());
        break;
      }

      case cuda_learning::SignalingMessage::kIceCandidate: {
        const auto& req = msg.ice_candidate();
        std::string error;
        const bool success = manager_->HandleRemoteCandidate(
            req.session_id(), req.candidate().candidate(), req.candidate().sdp_mid(),
            req.candidate().sdp_mline_index(), &error);

        cuda_learning::SignalingMessage response;
        auto* resp = response.mutable_ice_candidate_response();
        resp->set_session_id(req.session_id());
        resp->set_accepted(success);
        if (!success) {
          if (error.empty()) {
            error = "failed to add ICE candidate";
          }
          resp->set_message(error);
        } else {
          resp->set_message("candidate accepted");
        }
        if (req.has_trace_context()) {
          *resp->mutable_trace_context() = req.trace_context();
        }

        {
          std::lock_guard<std::mutex> lock(stream_mutex);
          if (!stream_active.load()) {
            break;
          }
          if (!stream->Write(response)) {
            spdlog::warn("[WebRTC:{}] Failed to send ice_candidate response",
                         req.session_id());
            stream_active = false;
            break;
          }
        }

        spdlog::info("[WebRTC:{}] ICE candidate processed via stream", req.session_id());
        break;
      }

      case cuda_learning::SignalingMessage::kCloseSession: {
        const auto& req = msg.close_session();
        std::string error;
        const bool success = manager_->CloseSession(req.session_id(), &error);

        cuda_learning::SignalingMessage response;
        auto* resp = response.mutable_close_session_response();
        resp->set_session_id(req.session_id());
        resp->set_closed(success);
        if (req.has_trace_context()) {
          *resp->mutable_trace_context() = req.trace_context();
        }

        {
          std::lock_guard<std::mutex> lock(stream_mutex);
          if (!stream_active.load()) {
            break;
          }
          if (!stream->Write(response)) {
            spdlog::warn("[WebRTC:{}] Failed to send close_session response",
                         req.session_id());
            stream_active = false;
            break;
          }
        }

        if (req.session_id() == current_session_id) {
          current_session_id.clear();
        }
        session_to_stream_map.erase(req.session_id());

        spdlog::info("[WebRTC:{}] Session closed via stream", req.session_id());
        stream_active = false;
        break;
      }

      case cuda_learning::SignalingMessage::kKeepAlive:
        spdlog::debug("Keepalive received, stream is alive");
        break;

      default:
        spdlog::warn("Unknown signaling message type: {}",
                     static_cast<int>(msg.message_case()));
        break;
    }
  }

  stream_active = false;

  if (!current_session_id.empty()) {
    std::string error;
    manager_->CloseSession(current_session_id, &error);
    spdlog::info("[WebRTC:{}] Session closed due to stream termination",
                 current_session_id);
  }

  candidate_sender.join();
  keepalive_thread.join();

  return ::grpc::Status::OK;
}

}  // namespace jrb::ports::grpc_service

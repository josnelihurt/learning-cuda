#include "cpp_accelerator/ports/grpc/webrtc_signaling_service_impl.h"

#include <string>

#include <spdlog/spdlog.h>

namespace jrb::ports::grpc_service {

WebRTCSignalingServiceImpl::WebRTCSignalingServiceImpl(std::shared_ptr<WebRTCManager> manager)
    : manager_(std::move(manager)) {}

::grpc::Status WebRTCSignalingServiceImpl::StartSession(
    ::grpc::ServerContext*, const cuda_learning::StartSessionRequest* request,
    cuda_learning::StartSessionResponse* response) {
  if (manager_ == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "WebRTC manager not available");
  }

  if (request == nullptr || response == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "invalid request");
  }

  if (request->session_id().empty() || request->sdp_offer().empty()) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "session_id and sdp_offer required");
  }

  std::string answer;
  std::string error;
  const bool success =
      manager_->CreateSession(request->session_id(), request->sdp_offer(), &answer, &error);
  if (!success) {
    if (error.empty()) {
      error = "failed to create WebRTC session";
    }
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, error);
  }

  response->set_session_id(request->session_id());
  response->set_sdp_answer(answer);
  if (request->has_trace_context()) {
    *response->mutable_trace_context() = request->trace_context();
  }

  spdlog::info("WebRTC signaling: session {} started", request->session_id());
  return ::grpc::Status::OK;
}

::grpc::Status WebRTCSignalingServiceImpl::SendIceCandidate(
    ::grpc::ServerContext*, const cuda_learning::SendIceCandidateRequest* request,
    cuda_learning::SendIceCandidateResponse* response) {
  if (manager_ == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "WebRTC manager not available");
  }

  if (request == nullptr || response == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "invalid request");
  }

  if (request->session_id().empty() || !request->has_candidate()) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "session_id and candidate required");
  }

  const auto& candidate = request->candidate();
  std::string error;
  const bool success =
      manager_->HandleRemoteCandidate(request->session_id(), candidate.candidate(),
                                      candidate.sdp_mid(), candidate.sdp_mline_index(), &error);

  response->set_session_id(request->session_id());
  response->set_accepted(success);
  if (!success) {
    if (error.empty()) {
      error = "failed to add ICE candidate";
    }
    response->set_message(error);
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, error);
  }

  response->set_message("candidate accepted");
  if (request->has_trace_context()) {
    *response->mutable_trace_context() = request->trace_context();
  }

  spdlog::info("WebRTC signaling: ICE candidate received for session {}", request->session_id());
  return ::grpc::Status::OK;
}

::grpc::Status WebRTCSignalingServiceImpl::CloseSession(
    ::grpc::ServerContext*, const cuda_learning::CloseSessionRequest* request,
    cuda_learning::CloseSessionResponse* response) {
  if (manager_ == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "WebRTC manager not available");
  }

  if (request == nullptr || response == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "invalid request");
  }

  if (request->session_id().empty()) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "session_id required");
  }

  std::string error;
  const bool success = manager_->CloseSession(request->session_id(), &error);
  if (!success) {
    if (error.empty()) {
      error = "failed to close WebRTC session";
    }
    response->set_session_id(request->session_id());
    response->set_closed(false);
    if (request->has_trace_context()) {
      *response->mutable_trace_context() = request->trace_context();
    }
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, error);
  }

  response->set_session_id(request->session_id());
  response->set_closed(true);
  if (request->has_trace_context()) {
    *response->mutable_trace_context() = request->trace_context();
  }

  spdlog::info("WebRTC signaling: session {} closed", request->session_id());
  return ::grpc::Status::OK;
}

}  // namespace jrb::ports::grpc_service

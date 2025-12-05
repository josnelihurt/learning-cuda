#pragma once

#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <grpcpp/grpcpp.h>
#pragma GCC diagnostic pop

#include "cpp_accelerator/ports/grpc/webrtc_manager.h"
#include "proto/_virtual_imports/webrtc_signal_proto/webrtc_signal.grpc.pb.h"

namespace jrb::ports::grpc_service {

class WebRTCSignalingServiceImpl final : public cuda_learning::WebRTCSignalingService::Service {
public:
  explicit WebRTCSignalingServiceImpl(std::shared_ptr<WebRTCManager> manager);
  ~WebRTCSignalingServiceImpl() override = default;

  ::grpc::Status StartSession(::grpc::ServerContext* context,
                              const cuda_learning::StartSessionRequest* request,
                              cuda_learning::StartSessionResponse* response) override;

  ::grpc::Status SendIceCandidate(::grpc::ServerContext* context,
                                  const cuda_learning::SendIceCandidateRequest* request,
                                  cuda_learning::SendIceCandidateResponse* response) override;

  ::grpc::Status CloseSession(::grpc::ServerContext* context,
                              const cuda_learning::CloseSessionRequest* request,
                              cuda_learning::CloseSessionResponse* response) override;

private:
  std::shared_ptr<WebRTCManager> manager_;
};

}  // namespace jrb::ports::grpc_service

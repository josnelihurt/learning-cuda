#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>

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

  ::grpc::Status SignalingStream(
      ::grpc::ServerContext* context,
      ::grpc::ServerReaderWriter<cuda_learning::SignalingMessage,
                                 cuda_learning::SignalingMessage>* stream) override;

private:
  std::shared_ptr<WebRTCManager> manager_;
};

}  // namespace jrb::ports::grpc_service

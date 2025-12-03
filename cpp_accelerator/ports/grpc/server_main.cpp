#include <csignal>
#include <memory>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"

#include "cpp_accelerator/ports/grpc/image_processor_service_impl.h"
#include "cpp_accelerator/ports/grpc/processor_engine_adapter.h"
#include "cpp_accelerator/ports/grpc/webrtc_manager.h"
#include "cpp_accelerator/ports/grpc/webrtc_signaling_service_impl.h"
#include "cpp_accelerator/ports/shared_lib/processor_engine.h"

ABSL_FLAG(std::string, listen_addr, "0.0.0.0:60061",
          "Address for the ImageProcessorService gRPC server to bind to.");
ABSL_FLAG(int, cuda_device_id, 0, "CUDA device id to initialize on startup.");
ABSL_FLAG(int, max_message_mb, 64, "Maximum inbound/outbound gRPC message size in MiB.");

namespace {

std::unique_ptr<grpc::Server> server_instance;

void HandleSignal(int signum) {
  spdlog::warn("Received signal {} - shutting down gRPC server", signum);
  if (server_instance) {
    server_instance->Shutdown();
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  auto engine = std::make_shared<jrb::ports::shared_lib::ProcessorEngine>("grpc-transport");

  cuda_learning::InitRequest init_request;
  init_request.set_cuda_device_id(absl::GetFlag(FLAGS_cuda_device_id));
  cuda_learning::InitResponse init_response;
  if (!engine->Initialize(init_request, &init_response) || init_response.code() != 0) {
    spdlog::error("Failed to initialize processor engine: {}", init_response.message());
    return 1;
  }

  auto adapter = std::make_shared<jrb::ports::grpc_service::ProcessorEngineAdapter>(engine);
  auto service = std::make_unique<jrb::ports::grpc_service::ImageProcessorServiceImpl>(adapter);
  auto webrtc_manager = std::make_shared<jrb::ports::grpc_service::WebRTCManager>();
  if (!webrtc_manager->Initialize()) {
    spdlog::warn("WebRTCManager failed to initialize");
  } else {
    spdlog::info("WebRTCManager ready for signaling");
  }
  auto signaling_service =
      std::make_unique<jrb::ports::grpc_service::WebRTCSignalingServiceImpl>(webrtc_manager);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(absl::GetFlag(FLAGS_listen_addr), grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  builder.RegisterService(signaling_service.get());

  const int message_bytes = absl::GetFlag(FLAGS_max_message_mb) * 1024 * 1024;
  builder.SetMaxReceiveMessageSize(message_bytes);
  builder.SetMaxSendMessageSize(message_bytes);

  server_instance = builder.BuildAndStart();
  if (!server_instance) {
    spdlog::error("Failed to start gRPC server on {}", absl::GetFlag(FLAGS_listen_addr));
    return 1;
  }

  spdlog::info("ImageProcessorService gRPC server listening on {}",
               absl::GetFlag(FLAGS_listen_addr));

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  server_instance->Wait();
  spdlog::info("gRPC server exited gracefully");
  return 0;
}

#include "cpp_accelerator/ports/grpc/webrtc_manager.h"

#include <spdlog/spdlog.h>

namespace jrb::ports::grpc_service {

WebRTCManager::WebRTCManager() : initialized_(false) {}

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
    spdlog::info("Initializing WebRTCManager...");
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
  initialized_ = false;
  spdlog::info("WebRTCManager shut down successfully");
}

}  // namespace jrb::ports::grpc_service


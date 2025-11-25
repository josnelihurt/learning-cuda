#pragma once

#include <memory>
#include <string>
#include <functional>

namespace jrb::ports::grpc_service {

class WebRTCManager {
 public:
  WebRTCManager();
  ~WebRTCManager();

  bool Initialize();
  void Shutdown();

  bool IsInitialized() const { return initialized_; }

 private:
  bool initialized_;
};

}  // namespace jrb::ports::grpc_service


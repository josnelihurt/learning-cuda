#pragma once

namespace jrb::ports::control {

// Driving port — the entry point for an accelerator control connection.
// Concrete implementations (e.g. gRPC mTLS client) live in adapters/grpc_control/.
class IControlPort {
 public:
  virtual ~IControlPort() = default;

  // Blocks, reconnecting on transient failure. Returns when Stop() is called.
  virtual void Run() = 0;

  // Signals Run() to return.
  virtual void Stop() = 0;
};

}  // namespace jrb::ports::control

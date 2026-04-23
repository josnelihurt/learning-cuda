#pragma once

#include <atomic>
#include <functional>

namespace jrb::core {

class SignalHandler {
public:
  static SignalHandler& GetInstance();

  using ShutdownCallback = std::function<void()>;
  using CrashCallback = std::function<void(int)>;

  // Registers handlers for SIGINT/SIGTERM (graceful) and crash signals
  void Initialize(ShutdownCallback on_graceful_shutdown,
                  CrashCallback on_crash = nullptr);

  // Restores default signal handlers
  void Shutdown();

  SignalHandler(const SignalHandler&) = delete;
  SignalHandler& operator=(const SignalHandler&) = delete;
  SignalHandler(SignalHandler&&) = delete;
  SignalHandler& operator=(SignalHandler&&) = delete;

private:
  SignalHandler() = default;
  ~SignalHandler() = default;

  // Static handlers for std::signal registration
  static void HandleSignal(int signum);
  static void CrashHandler(int signum);

  ShutdownCallback shutdown_callback_;
  CrashCallback crash_callback_;
  std::atomic<bool> initialized_{false};

  // Pointer to singleton instance for static handler access
  static SignalHandler* instance;
};

}  // namespace jrb::core

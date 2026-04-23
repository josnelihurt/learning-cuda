#include "src/cpp_accelerator/core/signal_handler.h"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <cxxabi.h>
#include <execinfo.h>
#include <unistd.h>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop

namespace jrb::core {

SignalHandler* SignalHandler::instance = nullptr;

SignalHandler& SignalHandler::GetInstance() {
  static SignalHandler singleton;
  instance = &singleton;
  return singleton;
}

void SignalHandler::Initialize(ShutdownCallback on_graceful_shutdown,
                                CrashCallback on_crash) {
  if (initialized_.load()) {
    return;
  }

  shutdown_callback_ = std::move(on_graceful_shutdown);
  crash_callback_ = std::move(on_crash);

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);
  std::signal(SIGSEGV, CrashHandler);
  std::signal(SIGBUS, CrashHandler);
  std::signal(SIGABRT, CrashHandler);
  std::signal(SIGFPE, CrashHandler);
  std::signal(SIGILL, CrashHandler);

  initialized_.store(true);
}

void SignalHandler::Shutdown() {
  if (!initialized_.load()) {
    return;
  }

  std::signal(SIGINT, SIG_DFL);
  std::signal(SIGTERM, SIG_DFL);
  std::signal(SIGSEGV, SIG_DFL);
  std::signal(SIGBUS, SIG_DFL);
  std::signal(SIGABRT, SIG_DFL);
  std::signal(SIGFPE, SIG_DFL);
  std::signal(SIGILL, SIG_DFL);

  shutdown_callback_ = nullptr;
  crash_callback_ = nullptr;
  initialized_.store(false);
}

void SignalHandler::HandleSignal(int signum) {
  (void)signum;
  if (instance && instance->shutdown_callback_) {
    instance->shutdown_callback_();
  }
}

void SignalHandler::CrashHandler(int signum) {
  const char* sig_names[] = {"",      "",       "SIGINT", "SIGQUIT", "SIGILL",  "",
                              "SIGABRT", "",     "SIGFPE", "",        "",        "SIGSEGV",
                              "",      "",       "",       "SIGTERM", "",        "",
                              "",      "",       "",       "",        "",        "",
                              "",      "",       "",       "",        "",        "",
                              "",      "SIGUSR1","SIGUSR2"};
  const char* sig_name = (signum >= 0 && signum < 33) ? sig_names[signum] : "UNKNOWN";

  char header[256];
  int n = snprintf(header, sizeof(header),
                   "\n[CRASH] *** Signal %s (%d) received — dumping backtrace ***\n",
                   sig_name, signum);
  write(STDERR_FILENO, header, static_cast<size_t>(n));

  void* callstack[64];
  int frames = backtrace(callstack, 64);
  backtrace_symbols_fd(callstack, frames, STDERR_FILENO);

  const char* footer = "[CRASH] *** End of backtrace — re-raising for core dump ***\n\n";
  write(STDERR_FILENO, footer, strlen(footer));

  if (instance && instance->crash_callback_) {
    instance->crash_callback_(signum);
  }

  spdlog::default_logger()->flush();

  signal(signum, SIG_DFL);
  raise(signum);
}

}  // namespace jrb::core

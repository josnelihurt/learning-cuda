#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <rtc/rtc.hpp>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#include "src/cpp_accelerator/adapters/webrtc/webrtc_session_state.h"

namespace jrb::application::engine {
class ProcessorEngine;
}

namespace jrb::domain::interfaces {
class IImageSink;
}

namespace jrb::adapters::camera {
class CameraHub;
}

namespace jrb::application::server_info {
class IServerInfoProvider;
}

namespace jrb::adapters::webrtc {

class ControlMessageDispatcher {
public:
  struct Config {
    jrb::application::server_info::IServerInfoProvider* server_info;
    std::shared_ptr<jrb::application::engine::ProcessorEngine> engine;
    std::shared_ptr<jrb::adapters::camera::CameraHub> camera_hub;
    std::shared_ptr<jrb::domain::interfaces::IImageSink> image_sink;
    std::string device_id;
    std::string display_name;
    std::string captures_dir;
    std::string accelerator_version;
    // Wraps the manager's camera frame routing — constructed in Initialize() via lambda.
    // Signature omits SessionState because the manager resolves the session internally.
    std::function<void(const std::string&, rtc::binary, rtc::FrameInfo)> video_frame_handler;
  };

  explicit ControlMessageDispatcher(Config config);

  void Dispatch(const cuda_learning::ControlRequest& request, const std::string& session_id,
                SessionState& state, cuda_learning::ControlResponse* response);

private:
  void HandleListFilters(const cuda_learning::ControlRequest& request,
                         cuda_learning::ControlResponse* response);

  void HandleGetVersion(const cuda_learning::ControlRequest& request,
                        cuda_learning::ControlResponse* response);

  void HandleGetAcceleratorCapabilities(const cuda_learning::ControlRequest& request,
                                        cuda_learning::ControlResponse* response);

  void HandleStartCameraStream(const cuda_learning::ControlRequest& request,
                               const std::string& session_id, SessionState& state,
                               cuda_learning::ControlResponse* response);

  void HandleStopCameraStream(const std::string& session_id, SessionState& state,
                              cuda_learning::ControlResponse* response);

  void HandleCaptureFrame(const std::string& session_id, SessionState& state,
                          cuda_learning::ControlResponse* response);

  void HandleListCapturedImages(const cuda_learning::ControlRequest& request,
                                const std::string& session_id,
                                cuda_learning::ControlResponse* response);

  void HandleGetCapturedImage(const cuda_learning::ControlRequest& request,
                              const std::string& session_id,
                              cuda_learning::ControlResponse* response);

  void HandleDeleteCapturedImage(const cuda_learning::ControlRequest& request,
                                 const std::string& session_id,
                                 cuda_learning::ControlResponse* response);

  using HandlerFn = std::function<void(const cuda_learning::ControlRequest&, const std::string&,
                                       SessionState&, cuda_learning::ControlResponse*)>;
  std::unordered_map<int, HandlerFn> handlers_;

  jrb::application::server_info::IServerInfoProvider* server_info_;
  std::shared_ptr<jrb::application::engine::ProcessorEngine> engine_;
  std::shared_ptr<jrb::adapters::camera::CameraHub> camera_hub_;
  std::shared_ptr<jrb::domain::interfaces::IImageSink> image_sink_;
  std::string device_id_;
  std::string display_name_;
  std::string captures_dir_;
  std::string accelerator_version_;
  std::function<void(const std::string&, rtc::binary, rtc::FrameInfo)> video_frame_handler_;
};

}  // namespace jrb::adapters::webrtc

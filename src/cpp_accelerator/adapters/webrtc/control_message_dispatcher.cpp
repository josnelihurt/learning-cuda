#include "src/cpp_accelerator/adapters/webrtc/control_message_dispatcher.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "src/cpp_accelerator/application/engine/processor_engine.h"
#include "src/cpp_accelerator/application/server_info/accelerator_label.h"
#include "src/cpp_accelerator/application/server_info/i_server_info_provider.h"
#include "src/cpp_accelerator/domain/interfaces/image_sink.h"

namespace jrb::adapters::webrtc {

using cuda_learning::AcceleratorType;
using cuda_learning::ControlRequest;
using cuda_learning::ControlResponse;
using cuda_learning::GetCapabilitiesResponse;
using cuda_learning::GetVersionInfoResponse;
using cuda_learning::ListFiltersResponse;

ControlMessageDispatcher::ControlMessageDispatcher(Config config)
    : server_info_(config.server_info),
      engine_(std::move(config.engine)),
      camera_hub_(std::move(config.camera_hub)),
      image_sink_(std::move(config.image_sink)),
      device_id_(std::move(config.device_id)),
      display_name_(std::move(config.display_name)),
      captures_dir_(std::move(config.captures_dir)),
      video_frame_handler_(std::move(config.video_frame_handler)) {
  handlers_[ControlRequest::kListFilters] =
      [this](const auto& req, const auto& /*sid*/, auto& /*state*/, auto* resp) {
        HandleListFilters(req, resp);
      };
  handlers_[ControlRequest::kGetVersion] =
      [this](const auto& req, const auto& /*sid*/, auto& /*state*/, auto* resp) {
        HandleGetVersion(req, resp);
      };
  handlers_[ControlRequest::kGetAcceleratorCapabilities] =
      [this](const auto& req, const auto& /*sid*/, auto& /*state*/, auto* resp) {
        HandleGetAcceleratorCapabilities(req, resp);
      };
  handlers_[ControlRequest::kStartCameraStream] =
      [this](const auto& req, const auto& sid, auto& state, auto* resp) {
        HandleStartCameraStream(req, sid, state, resp);
      };
  handlers_[ControlRequest::kStopCameraStream] =
      [this](const auto& /*req*/, const auto& sid, auto& state, auto* resp) {
        HandleStopCameraStream(sid, state, resp);
      };
  handlers_[ControlRequest::kCaptureFrame] =
      [this](const auto& /*req*/, const auto& sid, auto& state, auto* resp) {
        HandleCaptureFrame(sid, state, resp);
      };
  handlers_[ControlRequest::kListCapturedImages] =
      [this](const auto& req, const auto& sid, auto& /*state*/, auto* resp) {
        HandleListCapturedImages(req, sid, resp);
      };
  handlers_[ControlRequest::kGetCapturedImage] =
      [this](const auto& req, const auto& sid, auto& /*state*/, auto* resp) {
        HandleGetCapturedImage(req, sid, resp);
      };
  handlers_[ControlRequest::kDeleteCapturedImage] =
      [this](const auto& req, const auto& sid, auto& /*state*/, auto* resp) {
        HandleDeleteCapturedImage(req, sid, resp);
      };
}

void ControlMessageDispatcher::Dispatch(const ControlRequest& request,
                                        const std::string& session_id,
                                        SessionState& state,
                                        ControlResponse* response) {
  auto it = handlers_.find(static_cast<int>(request.payload_case()));
  if (it == handlers_.end()) {
    auto* err = response->mutable_error();
    err->set_code("UNKNOWN_PAYLOAD");
    err->set_message("control request payload not recognized");
    spdlog::warn("[WebRTC:{}] Control: unknown payload case {}", session_id,
                 static_cast<int>(request.payload_case()));
    return;
  }
  it->second(request, session_id, state, response);
}

void ControlMessageDispatcher::HandleListFilters(const ControlRequest& request,
                                                 ControlResponse* response) {
  ListFiltersResponse list_resp;
  server_info_->PopulateListFiltersResponse(request.list_filters(), &list_resp);
  *response->mutable_list_filters() = std::move(list_resp);
}

void ControlMessageDispatcher::HandleGetVersion(const ControlRequest& request,
                                                ControlResponse* response) {
  GetVersionInfoResponse ver_resp;
  server_info_->PopulateVersionResponse(&ver_resp);
  ver_resp.set_api_version(request.get_version().api_version());
  *response->mutable_get_version() = std::move(ver_resp);
}

void ControlMessageDispatcher::HandleGetAcceleratorCapabilities(
    const ControlRequest& /*request*/, ControlResponse* response) {
  auto* caps_resp = response->mutable_get_accelerator_capabilities();
  caps_resp->set_device_id(device_id_);
  caps_resp->set_display_name(display_name_);

  std::set<AcceleratorType> accelerator_types;
  GetCapabilitiesResponse engine_caps;
  if (engine_ && engine_->GetCapabilities(&engine_caps)) {
    for (const auto& filter : engine_caps.capabilities().filters()) {
      for (const auto accelerator : filter.supported_accelerators()) {
        accelerator_types.insert(static_cast<AcceleratorType>(accelerator));
      }
    }
  }
  for (const auto type : accelerator_types) {
    auto* option = caps_resp->add_supported_options();
    option->set_type(type);
    option->set_label(jrb::application::server_info::AcceleratorTypeLabel(type));
  }
}

void ControlMessageDispatcher::HandleStartCameraStream(const ControlRequest& request,
                                                       const std::string& session_id,
                                                       SessionState& state,
                                                       ControlResponse* response) {
  const auto& req = request.start_camera_stream();
  spdlog::info("[WebRTC:{}] StartCameraStream sensor_id={} {}x{}@{}fps", session_id,
               req.sensor_id(), req.width(), req.height(), req.fps());
  auto* resp = response->mutable_start_camera_stream();
  if (!camera_hub_) {
    resp->set_accepted(false);
    resp->set_reason("camera hub unavailable");
    spdlog::error("[WebRTC:{}] StartCameraStream: camera hub unavailable", session_id);
    return;
  }
  state.camera_subscription.Reset();

  // The video_frame_handler_ resolves the session internally via WebRTCManager::GetSession,
  // so the camera hub callback only needs to forward session_id + frame data.
  auto frame_cb = [handler = video_frame_handler_,
                   sid = session_id](const rtc::binary& data, const rtc::FrameInfo& info) {
    handler(sid, data, info);
  };

  std::string cam_err;
  auto subscription = camera_hub_->Subscribe(req.sensor_id(), req.width(), req.height(),
                                             req.fps(), std::move(frame_cb), &cam_err);
  const bool ok = subscription.IsActive();
  resp->set_accepted(ok);
  if (!ok) {
    resp->set_reason(cam_err);
    spdlog::error("[WebRTC:{}] Failed to subscribe to sensor_id={}: {}", session_id,
                  req.sensor_id(), cam_err);
  } else {
    state.camera_subscription = std::move(subscription);
  }
}

void ControlMessageDispatcher::HandleStopCameraStream(const std::string& session_id,
                                                      SessionState& state,
                                                      ControlResponse* response) {
  spdlog::info("[WebRTC:{}] StopCameraStream", session_id);
  state.camera_subscription.Reset();
  auto* resp = response->mutable_stop_camera_stream();
  resp->set_stopped(true);
}

void ControlMessageDispatcher::HandleCaptureFrame(const std::string& session_id,
                                                  SessionState& state,
                                                  ControlResponse* response) {
  auto* resp = response->mutable_capture_frame();
  if (state.live_video_processor == nullptr) {
    resp->set_captured(false);
    resp->set_reason("no active video processor for this session");
    spdlog::warn("[WebRTC:{}] CaptureFrame: no active live video processor", session_id);
    return;
  }
  auto now = std::chrono::system_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_info{};
  localtime_r(&t, &tm_info);
  char ts[32];
  std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_info);
  std::string filename =
      std::string("capture_") + ts + "_" + std::to_string(ms.count()) + ".bmp";
  std::error_code ec;
  std::filesystem::create_directories(captures_dir_, ec);
  if (ec) {
    resp->set_captured(false);
    resp->set_reason(std::string("failed to create captures directory: ") + ec.message());
    spdlog::error("[WebRTC:{}] CaptureFrame: mkdir {} failed: {}", session_id, captures_dir_,
                  ec.message());
    return;
  }
  std::string filepath = captures_dir_ + "/" + filename;
  state.live_video_processor->RequestCapture(filepath);
  resp->set_captured(true);
  resp->set_filename(filename);
  spdlog::info("[WebRTC:{}] CaptureFrame queued: {}", session_id, filename);
}

void ControlMessageDispatcher::HandleListCapturedImages(const ControlRequest& request,
                                                        const std::string& session_id,
                                                        ControlResponse* response) {
  auto* resp = response->mutable_list_captured_images();
  const auto& req = request.list_captured_images();
  const int page_size = req.page_size() > 0 ? req.page_size() : 20;
  const int page = req.page();

  std::vector<std::filesystem::path> files;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(captures_dir_, ec)) {
    if (entry.path().extension() == ".bmp") {
      files.push_back(entry.path());
    }
  }
  if (ec) {
    spdlog::warn("[WebRTC:{}] ListCapturedImages: directory_iterator error: {}", session_id,
                 ec.message());
  }
  std::sort(files.rbegin(), files.rend());

  const int total = static_cast<int>(files.size());
  const int start = page * page_size;
  resp->set_total_count(total);
  resp->set_has_more(start + page_size < total);

  const int end = std::min(start + page_size, total);
  for (int i = start; i < end; ++i) {
    auto* info = resp->add_images();
    info->set_id(files[static_cast<size_t>(i)].stem().string());
    info->set_filename(files[static_cast<size_t>(i)].filename().string());
    std::error_code mtime_ec;
    auto mtime = std::filesystem::last_write_time(files[static_cast<size_t>(i)], mtime_ec);
    if (!mtime_ec) {
      auto sctp = std::chrono::time_point_cast<std::chrono::milliseconds>(
          std::chrono::file_clock::to_sys(mtime));
      info->set_captured_at_ms(sctp.time_since_epoch().count());
    }
  }
  spdlog::info("[WebRTC:{}] ListCapturedImages page={} count={} total={}", session_id, page,
               end - start, total);
}

void ControlMessageDispatcher::HandleGetCapturedImage(const ControlRequest& request,
                                                      const std::string& session_id,
                                                      ControlResponse* response) {
  auto* resp = response->mutable_get_captured_image();
  const auto& req = request.get_captured_image();
  const std::string filepath = captures_dir_ + "/" + req.id() + ".bmp";

  if (req.format() == cuda_learning::CAPTURED_IMAGE_FORMAT_BMP) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) {
      resp->set_found(false);
      resp->set_reason("image not found: " + req.id());
      spdlog::warn("[WebRTC:{}] GetCapturedImage(BMP) not found: {}", session_id, req.id());
      return;
    }
    const std::string bytes((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    resp->set_found(true);
    resp->set_image_data(bytes);
    resp->set_format(cuda_learning::CAPTURED_IMAGE_FORMAT_BMP);
    spdlog::info("[WebRTC:{}] GetCapturedImage(BMP) {} {} bytes", session_id, req.id(),
                 bytes.size());
  } else {
    std::vector<uint8_t> img_data;
    int w = 0;
    int h = 0;
    if (!image_sink_->readAsPng(filepath.c_str(), &img_data, &w, &h, req.max_width(),
                                req.max_height())) {
      resp->set_found(false);
      resp->set_reason("image not found or conversion failed: " + req.id());
      spdlog::warn("[WebRTC:{}] GetCapturedImage(PNG) not found: {}", session_id, req.id());
      return;
    }
    resp->set_found(true);
    resp->set_image_data(img_data.data(), img_data.size());
    resp->set_width(w);
    resp->set_height(h);
    resp->set_format(cuda_learning::CAPTURED_IMAGE_FORMAT_PNG);
    spdlog::info("[WebRTC:{}] GetCapturedImage(PNG) {} {}x{} {} bytes", session_id, req.id(), w,
                 h, img_data.size());
  }
}

void ControlMessageDispatcher::HandleDeleteCapturedImage(const ControlRequest& request,
                                                         const std::string& session_id,
                                                         ControlResponse* response) {
  auto* resp = response->mutable_delete_captured_image();
  const auto& req = request.delete_captured_image();
  const std::string filepath = captures_dir_ + "/" + req.id() + ".bmp";
  std::error_code ec;
  const bool removed = std::filesystem::remove(filepath, ec);
  if (!removed || ec) {
    resp->set_deleted(false);
    resp->set_reason(ec ? ec.message() : "file not found: " + req.id());
    spdlog::warn("[WebRTC:{}] DeleteCapturedImage failed: {} \xe2\x80\x94 {}", session_id,
                 req.id(), ec ? ec.message() : "not found");
  } else {
    resp->set_deleted(true);
    spdlog::info("[WebRTC:{}] DeleteCapturedImage ok: {}", session_id, req.id());
  }
}

}  // namespace jrb::adapters::webrtc

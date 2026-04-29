#include "src/cpp_accelerator/adapters/webrtc/webrtc_protocol.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <future>
#include <sstream>
#include <string>
#include <thread>
#include <variant>

#include <spdlog/spdlog.h>

#include "src/cpp_accelerator/adapters/webrtc/data_channel_framing.h"
#include "src/cpp_accelerator/application/engine/processor_engine.h"

namespace jrb::adapters::webrtc::internal {

cuda_learning::GenericFilterParameterType ConvertParamType(const std::string& type) {
  if (type == "select")   return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_SELECT;
  if (type == "range")    return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_RANGE;
  if (type == "number")   return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_NUMBER;
  if (type == "checkbox") return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_CHECKBOX;
  if (type == "text")     return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_TEXT;
  return cuda_learning::GENERIC_FILTER_PARAMETER_TYPE_UNSPECIFIED;
}

void CopyValidationRulesFromMetadata(const cuda_learning::FilterParameter& source,
                                     cuda_learning::GenericFilterParameter* target) {
  if (target == nullptr) {
    return;
  }
  for (const auto& [key, value] : source.metadata()) {
    (*target->mutable_metadata())[key] = value;
    try {
      if (key == "required") {
        target->set_required(value == "true" || value == "1");
      } else if (key == "min") {
        target->set_min_value(std::stod(value));
      } else if (key == "max") {
        target->set_max_value(std::stod(value));
      } else if (key == "step") {
        target->set_step(std::stod(value));
      } else if (key == "min_items") {
        target->set_min_items(static_cast<uint32_t>(std::stoul(value)));
      } else if (key == "max_items") {
        target->set_max_items(static_cast<uint32_t>(std::stoul(value)));
      } else if (key == "pattern") {
        target->set_pattern(value);
      }
    } catch (const std::exception&) {
      // Keep metadata passthrough even if typed conversion fails.
    }
  }
}

void PopulateGetVersionResponse(jrb::application::engine::ProcessorEngine* engine,
                                cuda_learning::GetVersionInfoResponse* response) {
  if (response == nullptr) return;
  if (engine == nullptr) {
    response->set_code(6);
    response->set_message("engine unavailable");
    return;
  }
  std::string server_version;
  static const char* version_file_paths[] = {"src/cpp_accelerator/VERSION",
                                             "../cpp_accelerator/VERSION",
                                             "../../cpp_accelerator/VERSION", "./VERSION", nullptr};
  for (int i = 0; version_file_paths[i] != nullptr; ++i) {
    std::ifstream file(version_file_paths[i]);
    if (file.is_open()) {
      std::getline(file, server_version);
      if (!server_version.empty()) break;
    }
  }
  if (server_version.empty()) server_version = "unknown";
  response->set_server_version(server_version);

  cuda_learning::GetCapabilitiesResponse caps_response;
  if (engine->GetCapabilities(&caps_response)) {
    const auto& caps = caps_response.capabilities();
    response->set_library_version(caps.library_version());
    response->set_build_date(caps.build_date());
    response->set_build_commit(caps.build_commit());
  } else {
    response->set_library_version("unknown");
    response->set_build_date("unknown");
    response->set_build_commit("unknown");
  }
  response->set_code(0);
  response->set_message("OK");
}

void PopulateListFiltersResponse(jrb::application::engine::ProcessorEngine* engine,
                                 const cuda_learning::ListFiltersRequest& req,
                                 cuda_learning::ListFiltersResponse* resp) {
  resp->set_api_version(req.api_version());
  if (engine == nullptr) {
    return;
  }
  cuda_learning::GetCapabilitiesResponse caps;
  if (!engine->GetCapabilities(&caps)) {
    return;
  }
  for (const auto& filter : caps.capabilities().filters()) {
    auto* gf = resp->add_filters();
    gf->set_id(filter.id());
    gf->set_name(filter.name());
    for (const auto& param : filter.parameters()) {
      auto* gp = gf->add_parameters();
      gp->set_id(param.id());
      gp->set_name(param.name());
      gp->set_type(ConvertParamType(param.type()));
      gp->set_default_value(param.default_value());
      CopyValidationRulesFromMetadata(param, gp);
      for (const auto& opt : param.options()) {
        auto* go = gp->add_options();
        go->set_value(opt);
        go->set_label(opt);
      }
    }
    for (const auto acc : filter.supported_accelerators()) {
      gf->add_supported_accelerators(static_cast<cuda_learning::AcceleratorType>(acc));
    }
  }
}

bool IsGoVideoSession(const std::string& value) {
  return value.rfind(kGoVideoSessionPrefix, 0) == 0;
}

bool ShouldRegisterSessionChannel(const std::string& session_id, const std::string& label) {
  return !IsGoVideoSession(session_id) && !IsGoVideoSession(label);
}

std::string AcceleratorTypeLabel(cuda_learning::AcceleratorType type) {
  switch (type) {
    case cuda_learning::ACCELERATOR_TYPE_CUDA:   return "CUDA";
    case cuda_learning::ACCELERATOR_TYPE_CPU:    return "CPU";
    case cuda_learning::ACCELERATOR_TYPE_OPENCL: return "OpenCL";
    case cuda_learning::ACCELERATOR_TYPE_VULKAN: return "Vulkan";
    default:                                     return "Unknown";
  }
}

std::string NormalizeFilterId(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized;
}

std::optional<std::string> FirstGenericValue(
    const cuda_learning::GenericFilterParameterSelection& selection) {
  if (selection.values().empty()) {
    return std::nullopt;
  }
  return selection.values(0);
}

cuda_learning::GrayscaleType MapStringToGrayscaleType(const std::string& value) {
  const std::string normalized = NormalizeFilterId(value);
  if (normalized == "bt709")      return cuda_learning::GRAYSCALE_TYPE_BT709;
  if (normalized == "average")    return cuda_learning::GRAYSCALE_TYPE_AVERAGE;
  if (normalized == "lightness")  return cuda_learning::GRAYSCALE_TYPE_LIGHTNESS;
  if (normalized == "luminosity") return cuda_learning::GRAYSCALE_TYPE_LUMINOSITY;
  return cuda_learning::GRAYSCALE_TYPE_BT601;
}

cuda_learning::BorderMode MapStringToBorderMode(const std::string& value) {
  const std::string normalized = NormalizeFilterId(value);
  if (normalized == "clamp") return cuda_learning::BORDER_MODE_CLAMP;
  if (normalized == "wrap")  return cuda_learning::BORDER_MODE_WRAP;
  return cuda_learning::BORDER_MODE_REFLECT;
}

bool ParseBool(const std::string& value) {
  const std::string normalized = NormalizeFilterId(value);
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

void ResolveGenericSelectionsInPlace(cuda_learning::ProcessImageRequest* request) {
  if (request == nullptr || request->generic_filters_size() == 0) {
    return;
  }

  std::vector<cuda_learning::FilterType> filters;
  filters.reserve(static_cast<size_t>(request->filters_size()));
  for (const int filter : request->filters()) {
    filters.push_back(static_cast<cuda_learning::FilterType>(filter));
  }
  cuda_learning::GrayscaleType grayscale = request->grayscale_type();
  cuda_learning::GaussianBlurParameters blur_params = request->blur_params();
  bool has_blur_params = request->has_blur_params();

  filters.clear();
  for (const auto& selection : request->generic_filters()) {
    const std::string filter_id = NormalizeFilterId(selection.filter_id());
    if (filter_id.empty() || filter_id == "none") {
      filters.push_back(cuda_learning::FILTER_TYPE_NONE);
      continue;
    }

    if (filter_id == "grayscale") {
      filters.push_back(cuda_learning::FILTER_TYPE_GRAYSCALE);
      for (const auto& parameter : selection.parameters()) {
        if (NormalizeFilterId(parameter.parameter_id()) != "algorithm") {
          continue;
        }
        const auto value = FirstGenericValue(parameter);
        if (value.has_value()) {
          grayscale = MapStringToGrayscaleType(*value);
        }
      }
      continue;
    }

    if (filter_id == "blur") {
      filters.push_back(cuda_learning::FILTER_TYPE_BLUR);
      for (const auto& parameter : selection.parameters()) {
        const auto value = FirstGenericValue(parameter);
        if (!value.has_value()) {
          continue;
        }
        const std::string parameter_id = NormalizeFilterId(parameter.parameter_id());
        if (parameter_id == "kernel_size") {
          try {
            int parsed = std::stoi(*value);
            parsed = std::max(1, parsed);
            if (parsed % 2 == 0) {
              parsed += 1;
            }
            blur_params.set_kernel_size(parsed);
            has_blur_params = true;
          } catch (const std::exception&) {
            spdlog::warn("Ignoring invalid blur kernel_size value: {}", *value);
          }
          continue;
        }
        if (parameter_id == "sigma") {
          try {
            const float parsed = std::stof(*value);
            if (parsed >= 0.0F) {
              blur_params.set_sigma(parsed);
              has_blur_params = true;
            }
          } catch (const std::exception&) {
            spdlog::warn("Ignoring invalid blur sigma value: {}", *value);
          }
          continue;
        }
        if (parameter_id == "border_mode") {
          blur_params.set_border_mode(MapStringToBorderMode(*value));
          has_blur_params = true;
          continue;
        }
        if (parameter_id == "separable") {
          blur_params.set_separable(ParseBool(*value));
          has_blur_params = true;
        }
      }
      continue;
    }

    if (filter_id == "model_inference") {
      filters.push_back(cuda_learning::FILTER_TYPE_MODEL_INFERENCE);
      auto* model_params = request->mutable_model_params();
      if (model_params->model_id().empty()) {
        model_params->set_model_id("yolov10n");
      }
      if (model_params->confidence_threshold() <= 0.0F) {
        model_params->set_confidence_threshold(0.5F);
      }
      for (const auto& parameter : selection.parameters()) {
        const auto value = FirstGenericValue(parameter);
        if (!value.has_value()) {
          continue;
        }
        const std::string parameter_id = NormalizeFilterId(parameter.parameter_id());
        if (parameter_id == "model_id") {
          if (!value->empty()) {
            model_params->set_model_id(*value);
          }
        } else if (parameter_id == "confidence_threshold") {
          try {
            const float parsed = std::stof(*value);
            if (parsed > 0.0F) {
              model_params->set_confidence_threshold(parsed);
            }
          } catch (const std::exception&) {
            spdlog::warn("Ignoring invalid model confidence_threshold: {}", *value);
          }
        }
      }
      continue;
    }

    spdlog::warn("Ignoring unsupported generic filter in ProcessImageRequest: {}",
                 selection.filter_id());
  }

  request->clear_filters();
  for (const auto filter : filters) {
    request->add_filters(filter);
  }

  if (grayscale == cuda_learning::GRAYSCALE_TYPE_UNSPECIFIED) {
    grayscale = cuda_learning::GRAYSCALE_TYPE_BT601;
  }
  request->set_grayscale_type(grayscale);

  if (has_blur_params) {
    request->mutable_blur_params()->CopyFrom(blur_params);
  }
}

bool ParseDataChannelRequest(const std::vector<std::byte>& assembled,
                             cuda_learning::ProcessImageRequest* process_request,
                             bool* is_keepalive) {
  if (process_request == nullptr || is_keepalive == nullptr) {
    return false;
  }
  *is_keepalive = false;

  cuda_learning::DataChannelRequest envelope;
  if (envelope.ParseFromArray(assembled.data(), static_cast<int>(assembled.size()))) {
    if (envelope.has_keepalive()) {
      *is_keepalive = true;
      return true;
    }
    if (envelope.has_process_image()) {
      *process_request = envelope.process_image();
      return true;
    }
    return false;
  }

  // Compatibility path: accept legacy raw ProcessImageRequest payloads.
  if (process_request->ParseFromArray(assembled.data(), static_cast<int>(assembled.size()))) {
    spdlog::warn("[WebRTC] Received legacy raw ProcessImageRequest payload; migrate client to DataChannelRequest envelope");
    return true;
  }

  return false;
}

void SendFramed(rtc::DataChannel& dc, const std::string& payload, uint32_t message_id) {
  const auto span = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(payload.data()), payload.size());
  auto chunks = PackMessage(message_id, span);
  for (auto& chunk : chunks) {
    if (!dc.send(std::move(chunk))) {
      spdlog::error("[framing] Failed to send chunk for message_id={}", message_id);
      return;
    }
  }
}

void CopyProcessMetadata(const cuda_learning::ProcessImageRequest& request,
                         cuda_learning::ProcessImageResponse* response) {
  if (response == nullptr) {
    return;
  }
  response->set_api_version(request.api_version());
  response->mutable_trace_context()->CopyFrom(request.trace_context());
}

int64_t CurrentUnixTimeMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string NormalizeCodecName(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return normalized;
}

std::optional<OutboundVideoConfig> FindOutboundVideoConfig(const rtc::Description& offer) {
  for (int index = 0; index < offer.mediaCount(); ++index) {
    const auto entry = offer.media(index);
    if (!std::holds_alternative<const rtc::Description::Media*>(entry)) {
      continue;
    }
    const auto* media = std::get<const rtc::Description::Media*>(entry);
    if (media == nullptr || media->type() != "video" ||
        media->direction() != rtc::Description::Direction::RecvOnly) {
      continue;
    }
    for (const int payload_type : media->payloadTypes()) {
      const auto* rtp_map = media->rtpMap(payload_type);
      if (rtp_map == nullptr) {
        continue;
      }
      if (NormalizeCodecName(rtp_map->format) == "H264") {
        return OutboundVideoConfig{media->mid(), payload_type};
      }
    }
  }
  return std::nullopt;
}

uint32_t MakeSsrc(const std::string& session_id) {
  const uint32_t hash = static_cast<uint32_t>(std::hash<std::string>{}(session_id));
  return hash == 0 ? 1U : hash;
}

// Strips a=extmap lines from SDP to prevent malformed RTP header issues.
// Browsers include RTP header extensions (transport-cc, abs-send-time, etc.)
// that libdatachannel's RTP parser cannot handle, causing every incoming frame
// to be marked as malformed and dropped. Removing extmap from the offer
// causes the browser to omit header extensions from its RTP packets.
std::string StripRtpHeaderExtensions(const std::string& sdp) {
  std::string result;
  result.reserve(sdp.size());
  std::istringstream stream(sdp);
  std::string line;
  while (std::getline(stream, line)) {
    const std::string_view view(line);
    if (view.find("a=extmap:") != std::string_view::npos ||
        view.find("a=extmap-allow-mixed") != std::string_view::npos) {
      continue;
    }
    result += line;
    result += "\r\n";
  }
  return result;
}

std::string BuildManualCandidateSdp(const std::string& session_id) {
  const char* public_ip_env = std::getenv("WEBRTC_PUBLIC_IP");
  const char* public_port_env = std::getenv("WEBRTC_PUBLIC_PORT");
  const char* public_tcp_port_env = std::getenv("WEBRTC_PUBLIC_TCP_PORT");

  if (public_ip_env == nullptr || public_port_env == nullptr) {
    spdlog::debug(
        "[WebRTC:{}] WEBRTC_PUBLIC_IP or WEBRTC_PUBLIC_PORT not set, skipping manual ICE candidate",
        session_id);
    return {};
  }

  try {
    const std::string public_ip = public_ip_env;
    const std::string public_port = public_port_env;
    std::ostringstream candidate_sdp;
    // SDP candidate format: candidate:foundation component transport priority address port typ type
    candidate_sdp << "a=candidate:1 1 UDP 2130706431 " << public_ip << " " << public_port
                  << " typ host\r\n";
    if (public_tcp_port_env != nullptr) {
      const std::string public_tcp_port = public_tcp_port_env;
      candidate_sdp << "a=candidate:2 1 TCP 2130706430 " << public_ip << " " << public_tcp_port
                    << " typ host tcptype passive\r\n";
      spdlog::info("[WebRTC:{}] Will inject TCP ICE candidate for firewall fallback: {}:{}",
                   session_id, public_ip, public_tcp_port);
    }
    spdlog::info("[WebRTC:{}] Will inject manual ICE candidate in SDP: {}:{}", session_id,
                 public_ip, public_port);
    return candidate_sdp.str();
  } catch (const std::exception& e) {
    spdlog::warn("[WebRTC:{}] Failed to prepare manual ICE candidate: {}", session_id, e.what());
    return {};
  }
}

bool WaitForSdpAnswer(const std::string& session_id,
                      std::shared_future<std::string> answer_future,
                      rtc::PeerConnection& pc,
                      std::string* sdp_answer_str) {
  const auto timeout = std::chrono::seconds(10);
  const auto start = std::chrono::steady_clock::now();

  // Check if the answer is already available synchronously.
  if (pc.localDescription().has_value()) {
    auto local_desc = pc.localDescription().value();
    if (local_desc.type() == rtc::Description::Type::Answer) {
      std::string sdp = local_desc.generateSdp();
      if (sdp_answer_str != nullptr && sdp_answer_str->empty()) {
        *sdp_answer_str = sdp;
        spdlog::info("[WebRTC:{}] SDP answer available immediately (length: {})", session_id,
                     sdp.length());
        return true;
      }
    }
  }

  spdlog::info("[WebRTC:{}] Waiting for SDP answer (timeout: {}s)", session_id, timeout.count());
  while ((std::chrono::steady_clock::now() - start) < timeout) {
    const auto status = answer_future.wait_for(std::chrono::milliseconds(100));
    if (status == std::future_status::ready) {
      try {
        const std::string answer = answer_future.get();
        if (sdp_answer_str != nullptr && sdp_answer_str->empty()) {
          *sdp_answer_str = answer;
        }
        spdlog::info("[WebRTC:{}] SDP answer received via callback (length: {})", session_id,
                     answer.length());
        return true;
      } catch (const std::exception& e) {
        spdlog::error("[WebRTC:{}] Error getting answer from future: {}", session_id, e.what());
      }
    }

    if (pc.localDescription().has_value()) {
      auto local_desc = pc.localDescription().value();
      if (local_desc.type() == rtc::Description::Type::Answer) {
        std::string sdp = local_desc.generateSdp();
        if (sdp_answer_str != nullptr && sdp_answer_str->empty()) {
          *sdp_answer_str = sdp;
          spdlog::info("[WebRTC:{}] Retrieved SDP answer directly (length: {})", session_id,
                       sdp.length());
          return true;
        }
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  spdlog::error("[WebRTC:{}] Timeout waiting for SDP answer ({}s)", session_id, timeout.count());
  return false;
}

}  // namespace jrb::adapters::webrtc::internal

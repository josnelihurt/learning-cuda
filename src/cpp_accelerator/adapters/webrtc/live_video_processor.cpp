#include "src/cpp_accelerator/adapters/webrtc/live_video_processor.h"

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <spdlog/spdlog.h>

#include "src/cpp_accelerator/adapters/webrtc/protocol/filter_resolver.h"
#include "src/cpp_accelerator/application/engine/processor_engine.h"

namespace jrb::adapters::webrtc {

using cuda_learning::ACCELERATOR_TYPE_CUDA;
using cuda_learning::ACCELERATOR_TYPE_UNSPECIFIED;
using cuda_learning::DetectionFrame;
using cuda_learning::FILTER_TYPE_NONE;
using cuda_learning::FILTER_TYPE_UNSPECIFIED;
using cuda_learning::FilterType;
using cuda_learning::ProcessImageRequest;
using cuda_learning::ProcessImageResponse;
using jrb::adapters::webrtc::protocol::ResolveGenericSelectionsInPlace;
using jrb::domain::interfaces::IImageSink;

namespace {

constexpr int kTargetFps = 30;
constexpr int kDefaultBitrate = 2'500'000;

std::string AvErrorToString(int error_code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(error_code, buffer, sizeof(buffer));
  return std::string(buffer);
}

bool IsFilterNone(const FilterType filter) {
  return filter == FILTER_TYPE_NONE || filter == FILTER_TYPE_UNSPECIFIED;
}

}  // namespace

LiveVideoProcessor::LiveVideoProcessor(jrb::application::engine::ProcessorEngine* engine,
                                           void* cuda_memory_pool, IImageSink* image_sink)
    : engine_(engine),
      cuda_memory_pool_(cuda_memory_pool),
      image_sink_(image_sink),
      decoder_context_(nullptr),
      encoder_context_(nullptr),
      decode_to_rgb_context_(nullptr),
      rgb_to_yuv_context_(nullptr),
      decoded_frame_(nullptr),
      rgb_input_frame_(nullptr),
      rgb_output_frame_(nullptr),
      yuv_frame_(nullptr),
      decode_packet_(nullptr),
      encode_packet_(nullptr),
      frame_width_(0),
      frame_height_(0),
      input_pixel_format_(-1),
      next_pts_(0) {}

LiveVideoProcessor::~LiveVideoProcessor() {
  av_packet_free(&decode_packet_);
  av_packet_free(&encode_packet_);
  av_frame_free(&decoded_frame_);
  av_frame_free(&rgb_input_frame_);
  av_frame_free(&rgb_output_frame_);
  av_frame_free(&yuv_frame_);
  sws_freeContext(decode_to_rgb_context_);
  sws_freeContext(rgb_to_yuv_context_);
  avcodec_free_context(&decoder_context_);
  avcodec_free_context(&encoder_context_);
}

void LiveVideoProcessor::RequestCapture(std::string filepath) {
  {
    std::lock_guard<std::mutex> lk(capture_mutex_);
    capture_filepath_ = std::move(filepath);
  }
  capture_pending_.store(true);
}

bool LiveVideoProcessor::UpdateFilterState(const ProcessImageRequest& request,
                                           ProcessImageRequest* state,
                                           std::string* error_message) const {
  if (state == nullptr) {
    if (error_message != nullptr) {
      *error_message = "filter state target is required";
    }
    return false;
  }

  *state = request;
  state->clear_image_data();
  state->set_width(0);
  state->set_height(0);
  state->set_channels(0);

  // Resolve generic_filters into the typed filters[] field once, here, so
  // per-frame paths don't need to copy + re-resolve.
  ResolveGenericSelectionsInPlace(state);
  return true;
}

bool LiveVideoProcessor::ProcessAccessUnit(const rtc::binary& access_unit,
                                           const rtc::FrameInfo& frame_info,
                                           const ProcessImageRequest& state,
                                           std::vector<EncodedAccessUnit>* encoded_units,
                                           DetectionFrame* detection_frame,
                                           std::string* error_message) {
  if (encoded_units == nullptr) {
    if (error_message != nullptr) {
      *error_message = "encoded_units output is required";
    }
    return false;
  }

  encoded_units->clear();

  if (engine_ == nullptr) {
    if (error_message != nullptr) {
      *error_message = "processor engine unavailable";
    }
    return false;
  }

  if (access_unit.empty()) {
    return true;
  }

  if (!EnsureDecoder(error_message)) {
    return false;
  }

  av_packet_unref(decode_packet_);
  decode_packet_->data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(access_unit.data()));
  decode_packet_->size = static_cast<int>(access_unit.size());

  const int send_packet_result = avcodec_send_packet(decoder_context_, decode_packet_);
  if (send_packet_result < 0) {
    if (error_message != nullptr) {
      *error_message =
          "failed to submit H264 packet to decoder: " + AvErrorToString(send_packet_result);
    }
    return false;
  }

  std::vector<uint8_t> decoded_rgb_buffer;
  while (true) {
    const int receive_frame_result = avcodec_receive_frame(decoder_context_, decoded_frame_);
    if (receive_frame_result == AVERROR(EAGAIN) || receive_frame_result == AVERROR_EOF) {
      return true;
    }
    if (receive_frame_result < 0) {
      if (error_message != nullptr) {
        *error_message = "failed to decode H264 frame: " + AvErrorToString(receive_frame_result);
      }
      return false;
    }

    if (!EnsureFrameResources(decoded_frame_->width, decoded_frame_->height, decoded_frame_->format,
                              error_message)) {
      av_frame_unref(decoded_frame_);
      return false;
    }

    if (!CopyDecodedFrameToRgbBuffer(&decoded_rgb_buffer, error_message)) {
      av_frame_unref(decoded_frame_);
      return false;
    }

    std::vector<uint8_t> processed_rgb_buffer = decoded_rgb_buffer;
    ProcessImageRequest processing_request;
    if (HasActiveFilters(state)) {
      if (!BuildProcessingRequest(state, decoded_frame_->width, decoded_frame_->height,
                                  decoded_rgb_buffer, &processing_request)) {
        if (error_message != nullptr && error_message->empty()) {
          *error_message = "failed to build live processing request";
        }
        av_frame_unref(decoded_frame_);
        return false;
      }

      ProcessImageResponse processing_response;
      if (!engine_->ProcessImage(processing_request, &processing_response, cuda_memory_pool_) ||
          processing_response.code() != 0) {
        if (error_message != nullptr) {
          *error_message =
              "processor engine failed for live camera frame: " + processing_response.message();
        }
        av_frame_unref(decoded_frame_);
        return false;
      }

      if (!CopyResponseToRgbBuffer(processing_response, decoded_frame_->width,
                                   decoded_frame_->height, &processed_rgb_buffer, error_message)) {
        av_frame_unref(decoded_frame_);
        return false;
      }

      if (detection_frame != nullptr && processing_response.detections_size() > 0) {
        detection_frame->Clear();
        detection_frame->set_frame_id(static_cast<uint64_t>(next_pts_));
        detection_frame->set_image_width(decoded_frame_->width);
        detection_frame->set_image_height(decoded_frame_->height);
        for (const auto& d : processing_response.detections()) {
          *detection_frame->add_detections() = d;
        }
      }
    }

    if (capture_pending_.exchange(false)) {
      std::string path;
      {
        std::lock_guard<std::mutex> lk(capture_mutex_);
        path = capture_filepath_;
      }
      if (image_sink_ != nullptr && !path.empty()) {
        image_sink_->writeBmp(path.c_str(), processed_rgb_buffer.data(), decoded_frame_->width,
                              decoded_frame_->height, 3);
      }
    }

    if (!EncodeRgbBuffer(processed_rgb_buffer, frame_info, encoded_units, error_message)) {
      av_frame_unref(decoded_frame_);
      return false;
    }

    av_frame_unref(decoded_frame_);
  }
}

bool LiveVideoProcessor::EnsureDecoder(std::string* error_message) {
  if (decoder_context_ != nullptr) {
    return true;
  }

  if (decoded_frame_ == nullptr) {
    decoded_frame_ = av_frame_alloc();
  }
  if (decode_packet_ == nullptr) {
    decode_packet_ = av_packet_alloc();
  }
  if (decoded_frame_ == nullptr || decode_packet_ == nullptr) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate decoder frame/packet";
    }
    return false;
  }

  const AVCodec* decoder = avcodec_find_decoder(AV_CODEC_ID_H264);
  if (decoder == nullptr) {
    if (error_message != nullptr) {
      *error_message = "H264 decoder not available";
    }
    return false;
  }

  decoder_context_ = avcodec_alloc_context3(decoder);
  if (decoder_context_ == nullptr) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate H264 decoder context";
    }
    return false;
  }

  const int open_result = avcodec_open2(decoder_context_, decoder, nullptr);
  if (open_result < 0) {
    if (error_message != nullptr) {
      *error_message = "failed to open H264 decoder: " + AvErrorToString(open_result);
    }
    return false;
  }

  return true;
}

bool LiveVideoProcessor::EnsureFrameResources(int width, int height, int input_format,
                                              std::string* error_message) {
  if (width <= 0 || height <= 0) {
    if (error_message != nullptr) {
      *error_message = "decoded frame has invalid dimensions";
    }
    return false;
  }

  const bool needs_recreate =
      width != frame_width_ || height != frame_height_ || input_format != input_pixel_format_;
  if (!needs_recreate) {
    return true;
  }

  sws_freeContext(decode_to_rgb_context_);
  sws_freeContext(rgb_to_yuv_context_);
  decode_to_rgb_context_ = nullptr;
  rgb_to_yuv_context_ = nullptr;

  if (rgb_input_frame_ == nullptr) {
    rgb_input_frame_ = av_frame_alloc();
  }
  if (rgb_output_frame_ == nullptr) {
    rgb_output_frame_ = av_frame_alloc();
  }
  if (yuv_frame_ == nullptr) {
    yuv_frame_ = av_frame_alloc();
  }
  if (rgb_input_frame_ == nullptr || rgb_output_frame_ == nullptr || yuv_frame_ == nullptr) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate scaler frames";
    }
    return false;
  }

  av_frame_unref(rgb_input_frame_);
  av_frame_unref(rgb_output_frame_);
  av_frame_unref(yuv_frame_);

  rgb_input_frame_->format = AV_PIX_FMT_RGB24;
  rgb_input_frame_->width = width;
  rgb_input_frame_->height = height;
  const int rgb_input_alloc = av_frame_get_buffer(rgb_input_frame_, 32);
  if (rgb_input_alloc < 0) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate RGB input frame: " + AvErrorToString(rgb_input_alloc);
    }
    return false;
  }

  rgb_output_frame_->format = AV_PIX_FMT_RGB24;
  rgb_output_frame_->width = width;
  rgb_output_frame_->height = height;
  const int rgb_output_alloc = av_frame_get_buffer(rgb_output_frame_, 32);
  if (rgb_output_alloc < 0) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate RGB output frame: " + AvErrorToString(rgb_output_alloc);
    }
    return false;
  }

  yuv_frame_->format = AV_PIX_FMT_YUV420P;
  yuv_frame_->width = width;
  yuv_frame_->height = height;
  const int yuv_alloc = av_frame_get_buffer(yuv_frame_, 32);
  if (yuv_alloc < 0) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate YUV frame: " + AvErrorToString(yuv_alloc);
    }
    return false;
  }

  decode_to_rgb_context_ =
      sws_getCachedContext(nullptr, width, height, static_cast<AVPixelFormat>(input_format), width,
                           height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (decode_to_rgb_context_ == nullptr) {
    if (error_message != nullptr) {
      *error_message = "failed to create decode-to-RGB scaler";
    }
    return false;
  }

  rgb_to_yuv_context_ =
      sws_getCachedContext(nullptr, width, height, AV_PIX_FMT_RGB24, width, height,
                           AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (rgb_to_yuv_context_ == nullptr) {
    if (error_message != nullptr) {
      *error_message = "failed to create RGB-to-YUV scaler";
    }
    return false;
  }

  if (!EnsureEncoder(width, height, error_message)) {
    return false;
  }

  frame_width_ = width;
  frame_height_ = height;
  input_pixel_format_ = input_format;
  return true;
}

bool LiveVideoProcessor::EnsureEncoder(int width, int height, std::string* error_message) {
  const bool needs_recreate = encoder_context_ == nullptr || encoder_context_->width != width ||
                              encoder_context_->height != height;
  if (!needs_recreate) {
    return true;
  }

  if (encode_packet_ == nullptr) {
    encode_packet_ = av_packet_alloc();
  }
  if (encode_packet_ == nullptr) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate encoder packet";
    }
    return false;
  }

  avcodec_free_context(&encoder_context_);

  const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
  if (encoder == nullptr) {
    if (error_message != nullptr) {
      *error_message = "H264 encoder not available";
    }
    return false;
  }

  encoder_context_ = avcodec_alloc_context3(encoder);
  if (encoder_context_ == nullptr) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate H264 encoder context";
    }
    return false;
  }

  encoder_context_->width = width;
  encoder_context_->height = height;
  encoder_context_->pix_fmt = AV_PIX_FMT_YUV420P;
  encoder_context_->time_base = AVRational{1, kTargetFps};
  encoder_context_->framerate = AVRational{kTargetFps, 1};
  encoder_context_->gop_size = kTargetFps;
  encoder_context_->max_b_frames = 0;
  encoder_context_->bit_rate = kDefaultBitrate;

  AVDictionary* encoder_options = nullptr;
  av_dict_set(&encoder_options, "preset", "ultrafast", 0);
  av_dict_set(&encoder_options, "tune", "zerolatency", 0);

  const int open_result = avcodec_open2(encoder_context_, encoder, &encoder_options);
  av_dict_free(&encoder_options);
  if (open_result < 0) {
    if (error_message != nullptr) {
      *error_message = "failed to open H264 encoder: " + AvErrorToString(open_result);
    }
    return false;
  }

  next_pts_ = 0;
  return true;
}

bool LiveVideoProcessor::BuildProcessingRequest(const ProcessImageRequest& state, int width,
                                                int height, const std::vector<uint8_t>& rgb_buffer,
                                                ProcessImageRequest* request) const {
  if (request == nullptr) {
    return false;
  }

  // `state` was already resolved by UpdateFilterState, so we just need to
  // attach the per-frame image data and dimensions.
  *request = state;
  request->set_image_data(rgb_buffer.data(), rgb_buffer.size());
  request->set_width(width);
  request->set_height(height);
  request->set_channels(3);

  if (request->accelerator() == ACCELERATOR_TYPE_UNSPECIFIED) {
    request->set_accelerator(ACCELERATOR_TYPE_CUDA);
  }
  return true;
}

bool LiveVideoProcessor::HasActiveFilters(const ProcessImageRequest& request) const {
  // `request` is the already-resolved live filter state — checking the typed
  // filters[] field is sufficient.
  for (const int filter : request.filters()) {
    if (!IsFilterNone(static_cast<FilterType>(filter))) {
      return true;
    }
  }
  return false;
}

bool LiveVideoProcessor::CopyDecodedFrameToRgbBuffer(std::vector<uint8_t>* rgb_buffer,
                                                     std::string* error_message) {
  if (rgb_buffer == nullptr) {
    if (error_message != nullptr) {
      *error_message = "rgb buffer target is required";
    }
    return false;
  }

  const int writable_result = av_frame_make_writable(rgb_input_frame_);
  if (writable_result < 0) {
    if (error_message != nullptr) {
      *error_message =
          "failed to make RGB input frame writable: " + AvErrorToString(writable_result);
    }
    return false;
  }

  const int scale_result =
      sws_scale(decode_to_rgb_context_, decoded_frame_->data, decoded_frame_->linesize, 0,
                decoded_frame_->height, rgb_input_frame_->data, rgb_input_frame_->linesize);
  if (scale_result <= 0) {
    if (error_message != nullptr) {
      *error_message = "failed to convert decoded frame to RGB";
    }
    return false;
  }

  rgb_buffer->assign(static_cast<size_t>(decoded_frame_->width) * decoded_frame_->height * 3, 0);
  for (int row = 0; row < decoded_frame_->height; ++row) {
    const auto* src = rgb_input_frame_->data[0] + row * rgb_input_frame_->linesize[0];
    auto* dst = rgb_buffer->data() + static_cast<size_t>(row) * decoded_frame_->width * 3;
    std::memcpy(dst, src, static_cast<size_t>(decoded_frame_->width) * 3);
  }

  return true;
}

bool LiveVideoProcessor::CopyResponseToRgbBuffer(const ProcessImageResponse& response, int width,
                                                 int height, std::vector<uint8_t>* rgb_buffer,
                                                 std::string* error_message) const {
  if (rgb_buffer == nullptr) {
    if (error_message != nullptr) {
      *error_message = "processed rgb buffer target is required";
    }
    return false;
  }

  if (response.width() != width || response.height() != height) {
    if (error_message != nullptr) {
      *error_message = "processor engine changed live frame dimensions unexpectedly";
    }
    return false;
  }

  const int channels = response.channels();
  const std::string& payload = response.image_data();
  if (channels <= 0 || payload.empty()) {
    if (error_message != nullptr) {
      *error_message = "processor engine returned an empty live frame";
    }
    return false;
  }

  rgb_buffer->assign(static_cast<size_t>(width) * height * 3, 0);
  if (channels == 1) {
    for (int index = 0; index < width * height; ++index) {
      const auto value = static_cast<uint8_t>(payload[static_cast<size_t>(index)]);
      (*rgb_buffer)[static_cast<size_t>(index) * 3] = value;
      (*rgb_buffer)[static_cast<size_t>(index) * 3 + 1] = value;
      (*rgb_buffer)[static_cast<size_t>(index) * 3 + 2] = value;
    }
    return true;
  }

  if (channels == 3) {
    std::memcpy(rgb_buffer->data(), payload.data(), rgb_buffer->size());
    return true;
  }

  if (channels == 4) {
    for (int index = 0; index < width * height; ++index) {
      const size_t src_offset = static_cast<size_t>(index) * 4;
      const size_t dst_offset = static_cast<size_t>(index) * 3;
      (*rgb_buffer)[dst_offset] = static_cast<uint8_t>(payload[src_offset]);
      (*rgb_buffer)[dst_offset + 1] = static_cast<uint8_t>(payload[src_offset + 1]);
      (*rgb_buffer)[dst_offset + 2] = static_cast<uint8_t>(payload[src_offset + 2]);
    }
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "processor engine returned unsupported channel count for live frame";
  }
  return false;
}

bool LiveVideoProcessor::FillRgbFrame(const std::vector<uint8_t>& rgb_buffer, AVFrame* frame,
                                      std::string* error_message) const {
  if (frame == nullptr) {
    if (error_message != nullptr) {
      *error_message = "target RGB frame is required";
    }
    return false;
  }

  const int writable_result = av_frame_make_writable(frame);
  if (writable_result < 0) {
    if (error_message != nullptr) {
      *error_message = "failed to make RGB frame writable: " + AvErrorToString(writable_result);
    }
    return false;
  }

  for (int row = 0; row < frame->height; ++row) {
    const auto* src =
        rgb_buffer.data() + static_cast<size_t>(row) * static_cast<size_t>(frame->width) * 3;
    auto* dst = frame->data[0] + row * frame->linesize[0];
    std::memcpy(dst, src, static_cast<size_t>(frame->width) * 3);
  }

  return true;
}

bool LiveVideoProcessor::EncodeRgbBuffer(const std::vector<uint8_t>& rgb_buffer,
                                         const rtc::FrameInfo& frame_info,
                                         std::vector<EncodedAccessUnit>* encoded_units,
                                         std::string* error_message) {
  if (!FillRgbFrame(rgb_buffer, rgb_output_frame_, error_message)) {
    return false;
  }

  const int writable_result = av_frame_make_writable(yuv_frame_);
  if (writable_result < 0) {
    if (error_message != nullptr) {
      *error_message = "failed to make encoder frame writable: " + AvErrorToString(writable_result);
    }
    return false;
  }

  const int scale_result =
      sws_scale(rgb_to_yuv_context_, rgb_output_frame_->data, rgb_output_frame_->linesize, 0,
                rgb_output_frame_->height, yuv_frame_->data, yuv_frame_->linesize);
  if (scale_result <= 0) {
    if (error_message != nullptr) {
      *error_message = "failed to convert processed RGB frame to YUV";
    }
    return false;
  }

  yuv_frame_->pts = next_pts_++;
  const int send_result = avcodec_send_frame(encoder_context_, yuv_frame_);
  if (send_result < 0) {
    if (error_message != nullptr) {
      *error_message = "failed to submit frame to H264 encoder: " + AvErrorToString(send_result);
    }
    return false;
  }

  while (true) {
    const int receive_result = avcodec_receive_packet(encoder_context_, encode_packet_);
    if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
      return true;
    }
    if (receive_result < 0) {
      if (error_message != nullptr) {
        *error_message =
            "failed to read H264 packet from encoder: " + AvErrorToString(receive_result);
      }
      return false;
    }

    EncodedAccessUnit encoded_unit{
        rtc::binary(
            reinterpret_cast<const std::byte*>(encode_packet_->data),
            reinterpret_cast<const std::byte*>(encode_packet_->data + encode_packet_->size)),
        frame_info.timestampSeconds.has_value() ? rtc::FrameInfo(*frame_info.timestampSeconds)
                                                : rtc::FrameInfo(frame_info.timestamp),
    };
    encoded_units->push_back(std::move(encoded_unit));
    av_packet_unref(encode_packet_);
  }
}

}  // namespace jrb::adapters::webrtc

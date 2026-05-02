#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rtc/rtc.hpp>

#include "proto/_virtual_imports/image_processor_service_proto/image_processor_service.pb.h"
#include "src/cpp_accelerator/domain/interfaces/image_sink.h"

namespace jrb::application::engine {
class ProcessorEngine;
}

struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace jrb::adapters::webrtc {

struct EncodedAccessUnit {
  rtc::binary data;
  rtc::FrameInfo frame_info;
};

class LiveVideoProcessor {
 public:
  LiveVideoProcessor(jrb::application::engine::ProcessorEngine* engine,
                     void* cuda_memory_pool,
                     jrb::domain::interfaces::IImageSink* image_sink);
  ~LiveVideoProcessor();

  bool UpdateFilterState(const cuda_learning::ProcessImageRequest& request,
                         cuda_learning::ProcessImageRequest* state,
                         std::string* error_message) const;

  bool ProcessAccessUnit(const rtc::binary& access_unit, const rtc::FrameInfo& frame_info,
                         const cuda_learning::ProcessImageRequest& state,
                         std::vector<EncodedAccessUnit>* encoded_units,
                         cuda_learning::DetectionFrame* detection_frame,
                         std::string* error_message);

  void RequestCapture(std::string filepath);

 private:
  bool EnsureDecoder(std::string* error_message);
  bool EnsureFrameResources(int width, int height, int input_format, std::string* error_message);
  bool EnsureEncoder(int width, int height, std::string* error_message);

  bool BuildProcessingRequest(const cuda_learning::ProcessImageRequest& state, int width, int height,
                              const std::vector<uint8_t>& rgb_buffer,
                              cuda_learning::ProcessImageRequest* request) const;
  bool HasActiveFilters(const cuda_learning::ProcessImageRequest& request) const;

  bool CopyDecodedFrameToRgbBuffer(std::vector<uint8_t>* rgb_buffer, std::string* error_message);
  bool CopyResponseToRgbBuffer(const cuda_learning::ProcessImageResponse& response, int width,
                               int height, std::vector<uint8_t>* rgb_buffer,
                               std::string* error_message) const;
  bool FillRgbFrame(const std::vector<uint8_t>& rgb_buffer, AVFrame* frame,
                    std::string* error_message) const;
  bool EncodeRgbBuffer(const std::vector<uint8_t>& rgb_buffer, const rtc::FrameInfo& frame_info,
                       std::vector<EncodedAccessUnit>* encoded_units,
                       std::string* error_message);

  jrb::application::engine::ProcessorEngine* engine_;
  void* cuda_memory_pool_;
  jrb::domain::interfaces::IImageSink* image_sink_;
  std::atomic<bool> capture_pending_{false};
  std::string capture_filepath_;
  std::mutex capture_mutex_;
  AVCodecContext* decoder_context_;
  AVCodecContext* encoder_context_;
  SwsContext* decode_to_rgb_context_;
  SwsContext* rgb_to_yuv_context_;
  AVFrame* decoded_frame_;
  AVFrame* rgb_input_frame_;
  AVFrame* rgb_output_frame_;
  AVFrame* yuv_frame_;
  AVPacket* decode_packet_;
  AVPacket* encode_packet_;
  int frame_width_;
  int frame_height_;
  int input_pixel_format_;
  int64_t next_pts_;
};

}  // namespace jrb::adapters::webrtc

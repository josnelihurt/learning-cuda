#include "src/cpp_accelerator/adapters/webrtc/data_channel_framing.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

namespace jrb::adapters::webrtc {

namespace {

// Envelope layout (little-endian, 17 bytes):
//   0-3   magic       "CKDC" = {0x43, 0x4B, 0x44, 0x43}
//   4     version     uint8, currently 1
//   5-8   message_id  uint32 LE
//   9-10  chunk_index uint16 LE, 0-based
//  11-12  chunk_count uint16 LE
//  13-16  payload_len uint32 LE, bytes in THIS chunk's payload
//  17+    payload

constexpr uint8_t kMagic[4] = {0x43, 0x4B, 0x44, 0x43};
constexpr uint8_t kVersion = 1;
constexpr size_t kHeaderSize = 17;

void WriteU32LE(std::byte* dst, uint32_t v) {
  dst[0] = static_cast<std::byte>(v & 0xffu);
  dst[1] = static_cast<std::byte>((v >> 8u) & 0xffu);
  dst[2] = static_cast<std::byte>((v >> 16u) & 0xffu);
  dst[3] = static_cast<std::byte>((v >> 24u) & 0xffu);
}

void WriteU16LE(std::byte* dst, uint16_t v) {
  dst[0] = static_cast<std::byte>(v & 0xffu);
  dst[1] = static_cast<std::byte>((v >> 8u) & 0xffu);
}

uint32_t ReadU32LE(const std::byte* src) {
  return static_cast<uint32_t>(src[0]) |
         (static_cast<uint32_t>(src[1]) << 8u) |
         (static_cast<uint32_t>(src[2]) << 16u) |
         (static_cast<uint32_t>(src[3]) << 24u);
}

uint16_t ReadU16LE(const std::byte* src) {
  return static_cast<uint16_t>(static_cast<uint16_t>(src[0]) |
                                (static_cast<uint16_t>(src[1]) << 8u));
}

}  // namespace

std::vector<std::vector<std::byte>> PackMessage(
    uint32_t message_id, std::span<const std::byte> payload) {
  if (payload.size() > kMaxMessageBytes) {
    throw std::invalid_argument(
        "PackMessage: payload too large (" + std::to_string(payload.size()) +
        " bytes), max is " + std::to_string(kMaxMessageBytes));
  }

  const uint32_t total = static_cast<uint32_t>(payload.size());
  const uint16_t chunk_count =
      total == 0u
          ? 1u
          : static_cast<uint16_t>((total + kChunkPayloadMax - 1u) / kChunkPayloadMax);

  std::vector<std::vector<std::byte>> chunks;
  chunks.reserve(chunk_count);

  for (uint16_t i = 0; i < chunk_count; ++i) {
    const uint32_t start = static_cast<uint32_t>(i) * kChunkPayloadMax;
    const uint32_t len =
        (total == 0u) ? 0u : std::min(kChunkPayloadMax, total - start);

    std::vector<std::byte> chunk(kHeaderSize + len);

    std::memcpy(chunk.data(), kMagic, 4);
    chunk[4] = static_cast<std::byte>(kVersion);
    WriteU32LE(chunk.data() + 5, message_id);
    WriteU16LE(chunk.data() + 9, i);
    WriteU16LE(chunk.data() + 11, chunk_count);
    WriteU32LE(chunk.data() + 13, len);
    if (len > 0u) {
      std::memcpy(chunk.data() + kHeaderSize, payload.data() + start, len);
    }

    chunks.push_back(std::move(chunk));
  }

  return chunks;
}

ChunkReassembler::ChunkReassembler(std::chrono::milliseconds timeout)
    : timeout_(timeout) {}

void ChunkReassembler::EvictTimedOut() {
  const auto now = std::chrono::steady_clock::now();
  std::vector<uint32_t> to_evict;

  for (const auto& [id, msg] : in_flight_) {
    if (now - msg.first_chunk_at > timeout_) {
      spdlog::warn("[framing] Timeout for message_id={}: received {}/{} chunks",
                   id, msg.received, msg.chunk_count);
      to_evict.push_back(id);
    }
  }

  for (const uint32_t id : to_evict) {
    in_flight_.erase(id);
    in_flight_order_.erase(
        std::remove(in_flight_order_.begin(), in_flight_order_.end(), id),
        in_flight_order_.end());
  }
}

void ChunkReassembler::EvictOldestIfFull() {
  if (static_cast<int>(in_flight_.size()) < kMaxInFlightMessages) {
    return;
  }
  const uint32_t oldest = in_flight_order_.front();
  in_flight_order_.erase(in_flight_order_.begin());
  in_flight_.erase(oldest);
  spdlog::warn("[framing] Evicted oldest in-flight message_id={} (cap={})",
               oldest, kMaxInFlightMessages);
}

std::optional<std::vector<std::byte>> ChunkReassembler::PushChunk(
    std::span<const std::byte> raw) {
  EvictTimedOut();

  if (raw.size() < kHeaderSize) {
    spdlog::warn("[framing] Chunk too small ({} bytes), dropping", raw.size());
    return std::nullopt;
  }

  if (std::memcmp(raw.data(), kMagic, 4) != 0) {
    spdlog::warn("[framing] Bad magic bytes, dropping chunk");
    return std::nullopt;
  }

  const uint8_t version = static_cast<uint8_t>(raw[4]);
  if (version != kVersion) {
    spdlog::warn("[framing] Unknown version {}, dropping chunk", version);
    return std::nullopt;
  }

  const uint32_t message_id = ReadU32LE(raw.data() + 5);
  const uint16_t chunk_index = ReadU16LE(raw.data() + 9);
  const uint16_t chunk_count = ReadU16LE(raw.data() + 11);
  const uint32_t payload_len = ReadU32LE(raw.data() + 13);

  if (chunk_count < 1u || chunk_count > kMaxChunks) {
    spdlog::warn("[framing] Invalid chunk_count={}, dropping", chunk_count);
    return std::nullopt;
  }

  if (kHeaderSize + payload_len > raw.size()) {
    spdlog::warn("[framing] payload_len={} exceeds buffer size={}, dropping",
                 payload_len, raw.size());
    return std::nullopt;
  }

  if (chunk_index >= chunk_count) {
    spdlog::warn("[framing] chunk_index={} >= chunk_count={}, dropping",
                 chunk_index, chunk_count);
    return std::nullopt;
  }

  auto it = in_flight_.find(message_id);
  if (it == in_flight_.end()) {
    EvictOldestIfFull();
    InFlightMessage msg;
    msg.chunk_count = chunk_count;
    msg.received = 0;
    msg.first_chunk_at = std::chrono::steady_clock::now();
    msg.chunks.resize(chunk_count);
    in_flight_.emplace(message_id, std::move(msg));
    in_flight_order_.push_back(message_id);
    it = in_flight_.find(message_id);
  }

  InFlightMessage& msg = it->second;

  if (msg.chunk_count != chunk_count) {
    spdlog::warn("[framing] chunk_count mismatch for message_id={}, dropping",
                 message_id);
    return std::nullopt;
  }

  if (msg.chunks[chunk_index].has_value()) {
    return std::nullopt;  // duplicate — ignore silently
  }

  const auto* payload_ptr = raw.data() + kHeaderSize;
  msg.chunks[chunk_index] =
      std::vector<std::byte>(payload_ptr, payload_ptr + payload_len);
  ++msg.received;

  if (msg.received < msg.chunk_count) {
    return std::nullopt;
  }

  // All chunks present — concatenate in order and return
  size_t total = 0;
  for (const auto& c : msg.chunks) {
    total += c->size();
  }
  std::vector<std::byte> result;
  result.reserve(total);
  for (const auto& c : msg.chunks) {
    result.insert(result.end(), c->begin(), c->end());
  }

  spdlog::debug("[framing] Message {} complete: {} bytes in {} chunks",
                message_id, total, chunk_count);

  in_flight_.erase(it);
  in_flight_order_.erase(
      std::remove(in_flight_order_.begin(), in_flight_order_.end(), message_id),
      in_flight_order_.end());

  return result;
}

}  // namespace jrb::adapters::webrtc

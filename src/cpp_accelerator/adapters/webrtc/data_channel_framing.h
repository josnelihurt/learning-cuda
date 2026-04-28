#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace jrb::adapters::webrtc {

// Envelope constants — must match the TypeScript implementation exactly.
inline constexpr uint32_t kChunkPayloadMax = 65519;   // 65536 - 17 (header size)
inline constexpr uint32_t kMaxMessageBytes = 16u * 1024u * 1024u;
inline constexpr uint16_t kMaxChunks = 257;            // ceil(16 MiB / kChunkPayloadMax)
inline constexpr int kChunkAssemblyTimeoutMs = 5000;
inline constexpr int kMaxInFlightMessages = 16;

// Split payload into one or more framed chunks ready for DataChannel.send().
// Each returned buffer is at most 65536 bytes (header + kChunkPayloadMax payload).
// Throws std::invalid_argument when payload exceeds kMaxMessageBytes.
std::vector<std::vector<std::byte>> PackMessage(
    uint32_t message_id, std::span<const std::byte> payload);

class ChunkReassembler {
 public:
  explicit ChunkReassembler(
      std::chrono::milliseconds timeout =
          std::chrono::milliseconds{kChunkAssemblyTimeoutMs});

  // Feed one raw envelope buffer received from the data channel.
  // Returns the reassembled payload when all chunks for a message_id are
  // received; std::nullopt otherwise (incomplete, duplicate, or error).
  std::optional<std::vector<std::byte>> PushChunk(std::span<const std::byte> raw);

 private:
  struct InFlightMessage {
    std::vector<std::optional<std::vector<std::byte>>> chunks;
    uint16_t chunk_count{};
    uint16_t received{};
    std::chrono::steady_clock::time_point first_chunk_at;
  };

  void EvictTimedOut();
  void EvictOldestIfFull();

  std::chrono::milliseconds timeout_;
  std::unordered_map<uint32_t, InFlightMessage> in_flight_;
  std::vector<uint32_t> in_flight_order_;  // oldest-first insertion order
};

}  // namespace jrb::adapters::webrtc

#include "src/cpp_accelerator/adapters/webrtc/data_channel_framing.h"

#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using namespace jrb::adapters::webrtc;

namespace {

std::vector<std::byte> MakePayload(size_t size,
                                    std::byte fill = std::byte{0xab}) {
  std::vector<std::byte> v(size);
  for (size_t i = 0; i < size; ++i) {
    v[i] = static_cast<std::byte>((i ^ static_cast<size_t>(fill)) & 0xffu);
  }
  return v;
}

std::optional<std::vector<std::byte>> RoundTrip(
    uint32_t msg_id, const std::vector<std::byte>& payload) {
  const auto chunks =
      PackMessage(msg_id, std::span<const std::byte>(payload));
  ChunkReassembler reassembler;
  std::optional<std::vector<std::byte>> result;
  for (const auto& chunk : chunks) {
    result = reassembler.PushChunk(std::span<const std::byte>(chunk));
  }
  return result;
}

}  // namespace

TEST(DataChannelFramingTest, Success_RoundTrip1Byte) {
  // Arrange
  const auto payload = MakePayload(1);
  // Act
  const auto result = RoundTrip(1, payload);
  // Assert
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, payload);
}

TEST(DataChannelFramingTest, Success_RoundTripChunkPayloadMaxMinus1) {
  const auto payload = MakePayload(kChunkPayloadMax - 1);
  const auto result = RoundTrip(1, payload);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, payload);
}

TEST(DataChannelFramingTest, Success_RoundTripChunkPayloadMax) {
  const auto payload = MakePayload(kChunkPayloadMax);
  const auto result = RoundTrip(1, payload);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, payload);
}

TEST(DataChannelFramingTest, Success_RoundTripChunkPayloadMaxPlus1) {
  const auto payload = MakePayload(kChunkPayloadMax + 1);
  const auto result = RoundTrip(1, payload);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, payload);
}

TEST(DataChannelFramingTest, Success_RoundTrip1MiB) {
  const auto payload = MakePayload(1 * 1024 * 1024);
  const auto result = RoundTrip(1, payload);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, payload);
}

TEST(DataChannelFramingTest, Success_RoundTrip16MiB) {
  const auto payload = MakePayload(16 * 1024 * 1024);
  const auto result = RoundTrip(1, payload);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, payload);
}

TEST(DataChannelFramingTest, Success_OutOfOrderChunkArrival) {
  // Arrange
  const size_t size = static_cast<size_t>(kChunkPayloadMax) * 3;
  const auto payload = MakePayload(size);
  const auto chunks =
      PackMessage(1, std::span<const std::byte>(payload));
  ASSERT_EQ(chunks.size(), 3u);

  // Act — deliver chunks out of order: 2, 0, 1
  ChunkReassembler sut;
  EXPECT_FALSE(sut.PushChunk(chunks[2]).has_value());
  EXPECT_FALSE(sut.PushChunk(chunks[0]).has_value());
  const auto result = sut.PushChunk(chunks[1]);

  // Assert
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, payload);
}

TEST(DataChannelFramingTest, Success_InterleavedMessages) {
  // Arrange
  const auto payload1 = MakePayload(kChunkPayloadMax * 2, std::byte{0x11});
  const auto payload2 = MakePayload(kChunkPayloadMax * 2, std::byte{0x22});
  const auto chunks1 =
      PackMessage(1, std::span<const std::byte>(payload1));
  const auto chunks2 =
      PackMessage(2, std::span<const std::byte>(payload2));
  ASSERT_EQ(chunks1.size(), 2u);
  ASSERT_EQ(chunks2.size(), 2u);

  // Act — interleave: msg1[0], msg2[0], msg1[1], msg2[1]
  ChunkReassembler sut;
  EXPECT_FALSE(sut.PushChunk(chunks1[0]).has_value());
  EXPECT_FALSE(sut.PushChunk(chunks2[0]).has_value());
  const auto result1 = sut.PushChunk(chunks1[1]);
  const auto result2 = sut.PushChunk(chunks2[1]);

  // Assert
  ASSERT_TRUE(result1.has_value());
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(*result1, payload1);
  EXPECT_EQ(*result2, payload2);
}

TEST(DataChannelFramingTest, Error_PayloadTooLargeThrows) {
  const auto payload = MakePayload(kMaxMessageBytes + 1);
  EXPECT_THROW(PackMessage(1, std::span<const std::byte>(payload)),
               std::invalid_argument);
}

TEST(DataChannelFramingTest, Error_CorruptedMagicDropsChunk) {
  // Arrange
  const auto payload = MakePayload(10);
  auto chunks = PackMessage(1, std::span<const std::byte>(payload));
  ASSERT_FALSE(chunks.empty());
  chunks[0][0] = std::byte{0x00};  // corrupt first magic byte

  // Act & Assert
  ChunkReassembler sut;
  EXPECT_FALSE(sut.PushChunk(std::span<const std::byte>(chunks[0])).has_value());
}

TEST(DataChannelFramingTest, Error_UnknownVersionDropsChunk) {
  // Arrange
  const auto payload = MakePayload(10);
  auto chunks = PackMessage(1, std::span<const std::byte>(payload));
  ASSERT_FALSE(chunks.empty());
  chunks[0][4] = std::byte{99};  // unknown version byte

  // Act & Assert
  ChunkReassembler sut;
  EXPECT_FALSE(sut.PushChunk(std::span<const std::byte>(chunks[0])).has_value());
}

TEST(DataChannelFramingTest, Error_BufferTooSmallDropsChunk) {
  const std::vector<std::byte> tiny(4, std::byte{0x00});
  ChunkReassembler sut;
  EXPECT_FALSE(sut.PushChunk(std::span<const std::byte>(tiny)).has_value());
}

TEST(DataChannelFramingTest, Error_TimeoutDiscardsPartialMessage) {
  // Arrange — use a 1 ms timeout so we don't have to sleep 5 seconds
  const size_t size = static_cast<size_t>(kChunkPayloadMax) * 2;
  const auto payload = MakePayload(size);
  const auto chunks =
      PackMessage(1, std::span<const std::byte>(payload));
  ASSERT_EQ(chunks.size(), 2u);

  ChunkReassembler sut{std::chrono::milliseconds{1}};

  // Act — push first chunk, let timeout expire, push second
  EXPECT_FALSE(sut.PushChunk(chunks[0]).has_value());
  std::this_thread::sleep_for(std::chrono::milliseconds{5});
  // The second chunk arrives after timeout. The evicted message is gone, so
  // a new entry is created with only chunk[1] present — still incomplete.
  EXPECT_FALSE(sut.PushChunk(chunks[1]).has_value());
}

TEST(DataChannelFramingTest, Edge_DuplicateChunkIgnored) {
  // Arrange
  const auto payload = MakePayload(kChunkPayloadMax * 2);
  const auto chunks =
      PackMessage(1, std::span<const std::byte>(payload));
  ASSERT_EQ(chunks.size(), 2u);

  // Act — send chunk[0] twice then chunk[1]
  ChunkReassembler sut;
  EXPECT_FALSE(sut.PushChunk(chunks[0]).has_value());
  EXPECT_FALSE(sut.PushChunk(chunks[0]).has_value());  // duplicate
  const auto result = sut.PushChunk(chunks[1]);

  // Assert
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, payload);
}

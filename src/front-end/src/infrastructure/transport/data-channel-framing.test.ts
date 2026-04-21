import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CHUNK_PAYLOAD_MAX, ChunkReassembler, packMessage } from './data-channel-framing';

function makePayload(size: number, fill = 0xab): Uint8Array {
  const payload = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    payload[i] = (i ^ fill) & 0xff;
  }
  return payload;
}

function roundTrip(msgId: number, payload: Uint8Array): Uint8Array | null {
  const reassembler = new ChunkReassembler();
  const chunks = packMessage(msgId, payload);
  let result: Uint8Array | null = null;
  for (const chunk of chunks) {
    result = reassembler.pushChunk(chunk);
  }
  return result;
}

// Verify byte-for-byte equality for small payloads; for large payloads only
// check size + boundary bytes to avoid enumerating millions of Uint8Array
// integer keys in vitest's deep-equality serializer (causes OOM at 16 MiB).
function assertRoundTrip(payload: Uint8Array): void {
  const result = roundTrip(1, payload);
  expect(result).not.toBeNull();
  if (payload.byteLength <= CHUNK_PAYLOAD_MAX + 1) {
    expect(result).toEqual(payload);
  } else {
    expect(result!.byteLength).toBe(payload.byteLength);
    expect(result![0]).toBe(payload[0]);
    expect(result![payload.byteLength - 1]).toBe(payload[payload.byteLength - 1]);
  }
}

describe('packMessage / ChunkReassembler round-trips', () => {
  it.each([
    ['1 byte', 1],
    ['CHUNK_PAYLOAD_MAX - 1', CHUNK_PAYLOAD_MAX - 1],
    ['CHUNK_PAYLOAD_MAX', CHUNK_PAYLOAD_MAX],
    ['CHUNK_PAYLOAD_MAX + 1', CHUNK_PAYLOAD_MAX + 1],
    ['1 MiB', 1 * 1024 * 1024],
    ['16 MiB', 16 * 1024 * 1024],
  ])('Success_roundTrip_%s', (_label, size) => {
    assertRoundTrip(makePayload(size));
  });

  it('Success_emptyPayload', () => {
    const payload = new Uint8Array(0);
    const result = roundTrip(1, payload);
    expect(result).not.toBeNull();
    expect(result!.byteLength).toBe(0);
  });
});

describe('ChunkReassembler chunk ordering', () => {
  it('Success_outOfOrderChunkArrival', () => {
    const payload = makePayload(CHUNK_PAYLOAD_MAX * 3);
    const chunks = packMessage(1, payload);
    expect(chunks).toHaveLength(3);

    const sut = new ChunkReassembler();
    expect(sut.pushChunk(chunks[2])).toBeNull();
    expect(sut.pushChunk(chunks[0])).toBeNull();
    const result = sut.pushChunk(chunks[1]);
    expect(result).not.toBeNull();
    expect(result).toEqual(payload);
  });

  it('Success_interleavedMessages', () => {
    const payload1 = makePayload(CHUNK_PAYLOAD_MAX * 2, 0x11);
    const payload2 = makePayload(CHUNK_PAYLOAD_MAX * 2, 0x22);
    const chunks1 = packMessage(1, payload1);
    const chunks2 = packMessage(2, payload2);

    const sut = new ChunkReassembler();
    expect(sut.pushChunk(chunks1[0])).toBeNull();
    expect(sut.pushChunk(chunks2[0])).toBeNull();
    const result1 = sut.pushChunk(chunks1[1]);
    const result2 = sut.pushChunk(chunks2[1]);

    expect(result1).not.toBeNull();
    expect(result2).not.toBeNull();
    expect(result1!.byteLength).toBe(payload1.byteLength);
    expect(result2!.byteLength).toBe(payload2.byteLength);
    expect(result1![0]).toBe(payload1[0]);
    expect(result2![0]).toBe(payload2[0]);
  });

  it('Edge_duplicateChunkIgnored', () => {
    const payload = makePayload(CHUNK_PAYLOAD_MAX * 2);
    const chunks = packMessage(1, payload);
    const sut = new ChunkReassembler();
    expect(sut.pushChunk(chunks[0])).toBeNull();
    expect(sut.pushChunk(chunks[0])).toBeNull(); // duplicate
    const result = sut.pushChunk(chunks[1]);
    expect(result).not.toBeNull();
    expect(result).toEqual(payload);
  });
});

describe('ChunkReassembler error paths', () => {
  it('Error_oversizedMessageThrows', () => {
    const payload = makePayload(16 * 1024 * 1024 + 1);
    expect(() => packMessage(1, payload)).toThrow();
  });

  it('Error_corruptedMagicDropsChunk', () => {
    const payload = makePayload(10);
    const chunks = packMessage(1, payload);
    const corrupted = chunks[0].slice(0);
    new Uint8Array(corrupted)[0] = 0x00;
    const sut = new ChunkReassembler();
    expect(sut.pushChunk(corrupted)).toBeNull();
  });

  it('Error_unknownVersionDropsChunk', () => {
    const payload = makePayload(10);
    const chunks = packMessage(1, payload);
    const corrupted = chunks[0].slice(0);
    new DataView(corrupted).setUint8(4, 99);
    const sut = new ChunkReassembler();
    expect(sut.pushChunk(corrupted)).toBeNull();
  });

  it('Error_bufferTooSmallDropsChunk', () => {
    const tiny = new ArrayBuffer(4);
    const sut = new ChunkReassembler();
    expect(sut.pushChunk(tiny)).toBeNull();
  });
});

describe('ChunkReassembler timeout', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('Error_timeoutDiscardsPartialMessage', () => {
    const payload = makePayload(CHUNK_PAYLOAD_MAX * 2);
    const chunks = packMessage(1, payload);
    const sut = new ChunkReassembler();

    expect(sut.pushChunk(chunks[0])).toBeNull();

    // Advance past the 5 s assembly timeout
    vi.advanceTimersByTime(6_000);

    // Second chunk arrives after timeout: the in-flight message was discarded,
    // so a new incomplete entry is created — still missing chunk[0] → null
    expect(sut.pushChunk(chunks[1])).toBeNull();
  });
});

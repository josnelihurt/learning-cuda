// Envelope layout (little-endian, 17 bytes of overhead):
//   0-3  magic        "CKDC" (0x43 0x4B 0x44 0x43)
//   4    version      uint8, currently 1
//   5-8  message_id   uint32 LE, per-direction monotonic counter
//   9-10 chunk_index  uint16 LE, 0-based
//  11-12 chunk_count  uint16 LE, total chunks for this message_id
//  13-16 payload_len  uint32 LE, bytes in THIS chunk's payload
//  17+   payload      N bytes

const MAGIC_0 = 0x43;
const MAGIC_1 = 0x4b;
const MAGIC_2 = 0x44;
const MAGIC_3 = 0x43;

const VERSION = 1;
const HEADER_SIZE = 17;

export const CHUNK_PAYLOAD_MAX = 65519; // 65536 - HEADER_SIZE
const MAX_MESSAGE_BYTES = 16 * 1024 * 1024;
const MAX_CHUNKS = Math.ceil(MAX_MESSAGE_BYTES / CHUNK_PAYLOAD_MAX); // 257
const CHUNK_ASSEMBLY_TIMEOUT_MS = 5_000;
const MAX_IN_FLIGHT_MESSAGES = 16;

let _nextMessageId = 0;

export function nextMessageId(): number {
  _nextMessageId = (_nextMessageId + 1) >>> 0; // uint32 wrap-around
  return _nextMessageId;
}

export function packMessage(messageId: number, payload: Uint8Array): ArrayBuffer[] {
  if (payload.byteLength > MAX_MESSAGE_BYTES) {
    throw new Error(
      `Message too large: ${payload.byteLength} bytes exceeds limit of ${MAX_MESSAGE_BYTES}`
    );
  }

  const chunkCount = payload.byteLength === 0 ? 1 : Math.ceil(payload.byteLength / CHUNK_PAYLOAD_MAX);
  const chunks: ArrayBuffer[] = [];

  for (let i = 0; i < chunkCount; i++) {
    const start = i * CHUNK_PAYLOAD_MAX;
    const slice = payload.subarray(start, start + CHUNK_PAYLOAD_MAX);
    const buf = new ArrayBuffer(HEADER_SIZE + slice.byteLength);
    const view = new DataView(buf);
    const bytes = new Uint8Array(buf);

    bytes[0] = MAGIC_0;
    bytes[1] = MAGIC_1;
    bytes[2] = MAGIC_2;
    bytes[3] = MAGIC_3;
    view.setUint8(4, VERSION);
    view.setUint32(5, messageId, /* littleEndian= */ true);
    view.setUint16(9, i, true);
    view.setUint16(11, chunkCount, true);
    view.setUint32(13, slice.byteLength, true);
    bytes.set(slice, HEADER_SIZE);

    chunks.push(buf);
  }

  return chunks;
}

interface InFlightMessage {
  chunks: (Uint8Array | null)[];
  chunkCount: number;
  received: number;
  timer: ReturnType<typeof setTimeout>;
}

export class ChunkReassembler {
  private readonly inFlight = new Map<number, InFlightMessage>();
  private readonly inFlightOrder: number[] = []; // oldest first

  pushChunk(buffer: ArrayBuffer): Uint8Array | null {
    if (buffer.byteLength < HEADER_SIZE) {
      console.warn('[framing] Chunk too small, dropping');
      return null;
    }

    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    if (bytes[0] !== MAGIC_0 || bytes[1] !== MAGIC_1 || bytes[2] !== MAGIC_2 || bytes[3] !== MAGIC_3) {
      console.warn('[framing] Bad magic bytes, dropping chunk');
      return null;
    }

    const version = view.getUint8(4);
    if (version !== VERSION) {
      console.warn(`[framing] Unknown version ${version}, dropping chunk`);
      return null;
    }

    const messageId = view.getUint32(5, true);
    const chunkIndex = view.getUint16(9, true);
    const chunkCount = view.getUint16(11, true);
    const payloadLen = view.getUint32(13, true);

    if (chunkCount < 1 || chunkCount > MAX_CHUNKS) {
      console.warn(`[framing] Invalid chunk_count ${chunkCount}, dropping`);
      return null;
    }

    if (HEADER_SIZE + payloadLen > buffer.byteLength) {
      console.warn(`[framing] payload_len ${payloadLen} exceeds buffer size, dropping`);
      return null;
    }

    if (chunkIndex >= chunkCount) {
      console.warn(`[framing] chunk_index ${chunkIndex} >= chunk_count ${chunkCount}, dropping`);
      return null;
    }

    let msg = this.inFlight.get(messageId);

    if (!msg) {
      if (this.inFlightOrder.length >= MAX_IN_FLIGHT_MESSAGES) {
        const oldest = this.inFlightOrder.shift()!;
        const evicted = this.inFlight.get(oldest);
        if (evicted) {
          clearTimeout(evicted.timer);
          this.inFlight.delete(oldest);
          console.warn(`[framing] Evicted oldest in-flight message_id=${oldest} (cap reached)`);
        }
      }

      msg = {
        chunks: new Array<Uint8Array | null>(chunkCount).fill(null),
        chunkCount,
        received: 0,
        timer: setTimeout(() => {
          const m = this.inFlight.get(messageId);
          if (m) {
            this.inFlight.delete(messageId);
            const idx = this.inFlightOrder.indexOf(messageId);
            if (idx >= 0) {
              this.inFlightOrder.splice(idx, 1);
            }
            console.warn(
              `[framing] Timeout for message_id=${messageId}: received ${m.received}/${m.chunkCount} chunks`
            );
          }
        }, CHUNK_ASSEMBLY_TIMEOUT_MS),
      };
      this.inFlight.set(messageId, msg);
      this.inFlightOrder.push(messageId);
    }

    if (msg.chunkCount !== chunkCount) {
      console.warn(`[framing] chunk_count mismatch for message_id=${messageId}, dropping`);
      return null;
    }

    if (msg.chunks[chunkIndex] !== null) {
      return null; // duplicate — ignore silently
    }

    msg.chunks[chunkIndex] = new Uint8Array(buffer, HEADER_SIZE, payloadLen).slice();
    msg.received++;

    if (msg.received < msg.chunkCount) {
      return null;
    }

    clearTimeout(msg.timer);
    this.inFlight.delete(messageId);
    const orderIdx = this.inFlightOrder.indexOf(messageId);
    if (orderIdx >= 0) {
      this.inFlightOrder.splice(orderIdx, 1);
    }

    let total = 0;
    for (const c of msg.chunks) {
      total += c!.byteLength;
    }
    const result = new Uint8Array(total);
    let offset = 0;
    for (const c of msg.chunks) {
      result.set(c!, offset);
      offset += c!.byteLength;
    }
    return result;
  }
}

# Video Streaming Optimization

Research and POC tasks for improving video transport from current WebSocket + base64 PNG to more efficient methods.

## Current Implementation Analysis

**Status**: Working but inefficient
- Transport: WebSocket with base64-encoded PNG
- Overhead: ~33% from base64 encoding + PNG compression CPU cost
- Latency: Acceptable for learning, not production-ready
- Bandwidth: ~2-5 MB/s for 720p @ 30 FPS

**Bottlenecks**:
1. PNG encoding on every frame (CPU-intensive)
2. Base64 encoding overhead
3. No video compression (each frame independent)
4. No adaptive bitrate

## Research Phase: Create POCs

### POC 1: Connect-RPC Implementation (In Progress)

**Status**: Migrated from stdlib HTTP to Connect-RPC with multi-source support

#### Completed
- [x] Added ImageProcessorService to proto with ProcessImage and StreamProcessVideo RPCs
- [x] Implemented Connect-RPC server in `webserver/pkg/interfaces/connectrpc/`
- [x] Refactored main.go to clean App structure
- [x] Setup buf for code generation with Docker
- [x] Added HTTP annotations for REST-friendly endpoints
- [x] Modularized protobuf into separate service definitions (common, config_service, image_processor_service)
- [x] Implemented ListInputs endpoint with BDD test coverage
- [x] Created dynamic video grid component supporting up to 9 sources
- [x] Added drawer UI for source selection with real-time updates
- [x] Support per-source filter and resolution configuration

#### Pending
- [ ] Implement StreamProcessVideo bidirectional streaming (currently returns Unimplemented)
- [ ] Update frontend to use Connect-RPC instead of WebSocket for video streaming
- [ ] Benchmark latency vs current WebSocket
- [ ] Add grpc-web support for browser compatibility

**Note**: Using Connect-RPC instead of traditional gRPC-Gateway. Connect provides native HTTP/JSON support without needing a separate gateway.

### POC 2: Binary Transport (Quick Win) - Next Optimization

**Status**: Current implementation uses PNG base64 (working but inefficient)

**Why**: Eliminate base64 overhead and PNG encoding/decoding

**Current Flow:**
```
Browser → canvas.toDataURL('png') → base64 → WebSocket → 
Go decode PNG → RGBA raw → RPC → C++/CUDA → 
RGBA raw → encode PNG → base64 → WebSocket → Browser
```

**Proposed Flow:**
```
Browser → canvas.getImageData() → raw RGBA → WebSocket binary → 
RPC → C++/CUDA → raw RGBA → WebSocket binary → Browser
```

#### Tasks
- [ ] Modify WebSocket to accept binary frames instead of JSON
- [ ] Frontend: use canvas.getImageData().data (Uint8ClampedArray)
- [ ] Backend: remove PNG decode step in websocket_handler.go
- [ ] Remove PNG encode step (return raw bytes)
- [ ] Frontend: create ImageData and putImageData directly
- [ ] Benchmark performance improvement

**Expected Outcome**: 
- 40-50% bandwidth reduction
- 30% CPU reduction
- 100-150 FPS (vs current ~25-30 FPS)
- Simpler code (no encode/decode)

### POC 3: H.264/H.265 with NVENC

**Why**: Biggest performance gain, hardware-accelerated encoding

#### Tasks
- [ ] Research NVIDIA Video Codec SDK (NVENC)
- [ ] Study H.264 encoding parameters (bitrate, keyframe interval)
- [ ] Investigate FFmpeg integration vs direct NVENC API
- [ ] Check browser H.264 decoding support

#### Implementation Plan - Option A (FFmpeg)
- [ ] Add FFmpeg to Docker container
- [ ] Create CGO wrapper for FFmpeg encoding
- [ ] Pipeline: CUDA process → FFmpeg H.264 encode → stream
- [ ] Use HLS or MPEG-DASH for adaptive bitrate
- [ ] Test with video.js or native `<video>` tag

#### Implementation Plan - Option B (Direct NVENC)
- [ ] Download NVIDIA Video Codec SDK
- [ ] Link NVENC library in Bazel build
- [ ] Create encoder wrapper: raw frames → H.264 NAL units
- [ ] Stream NAL units via WebSocket
- [ ] Use Media Source Extensions (MSE) in browser

**Expected Outcome**: 10-50x bandwidth reduction, GPU encoding, professional-grade streaming

**Resources**:
- https://developer.nvidia.com/video-codec-sdk
- https://trac.ffmpeg.org/wiki/HWAccelIntro
- https://developer.mozilla.org/en-US/docs/Web/API/Media_Source_Extensions_API

### POC 4: WebRTC

**Why**: Production-grade, adaptive bitrate, low latency, P2P capable

#### Tasks
- [ ] Research WebRTC data channels vs media tracks
- [ ] Study STUN/TURN for NAT traversal
- [ ] Review Pion WebRTC library (Go)
- [ ] Understand signaling protocols (SDP offer/answer)

#### Implementation Plan
- [ ] Set up signaling server (WebSocket for SDP exchange)
- [ ] Implement Pion WebRTC peer in Go
- [ ] Use WebRTC data channels for control
- [ ] Use media tracks for video (encoded with NVENC)
- [ ] Handle ICE candidates for connection establishment
- [ ] Test with built-in browser WebRTC

**Expected Outcome**: Production-ready streaming, sub-100ms latency, adaptive quality

**Resources**:
- https://github.com/pion/webrtc
- https://webrtc.org/getting-started/overview
- https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API

### POC 5: WebCodecs API (Browser-side encoding)


**Expected Outcome**: Massive bandwidth reduction on upload, client hardware encoding

**Resources**:
- https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API
- https://web.dev/webcodecs/


## Metrics to Track

For each POC, measure:
- [ ] Bandwidth usage (MB/s)
- [ ] Latency (frame capture → display)
- [ ] CPU usage (server-side)
- [ ] GPU usage (encoding + processing)
- [ ] Frame drop rate
- [ ] Quality (PSNR/SSIM if applicable)

## Decision Criteria

Choose implementation based on:
1. **Learning Value**: Does it teach new concepts?
2. **Performance Gain**: Is the improvement significant?
3. **Implementation Effort**: Time to working POC?
4. **Production Readiness**: Can it scale?
5. **Browser Support**: Does it work everywhere?

## Notes

- Don't optimize prematurely - current solution works for learning
- Each POC should be in a separate branch
- Document findings in this file
- Can run multiple transports simultaneously (feature flag)
- Consider adding `/api/transport/select` endpoint to switch modes


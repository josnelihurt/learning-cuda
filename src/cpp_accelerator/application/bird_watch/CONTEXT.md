# BirdWatcher — Architecture Context & Implementation Roadmap

## Problem Statement

Store captures at full IMX477 sensor resolution (4056×3040, ~12.3 MP) while keeping
streaming + inference at a transport-viable resolution (720p). The original pipeline
encoded everything at one resolution and the BMP could never exceed the encode
resolution.

---

## Hardware Facts (Jetson Orin Nano, JetPack R36.4, IMX477)

| Fact | Detail |
|------|--------|
| No NVENC hardware | `/dev/v4l2-nvenc` absent; x264enc fallback (~15% CPU at 1080p30) |
| Full sensor mode | 4056×3040 @ 15 fps (sensor mode 0) |
| Binned mode | 2028×1520 @ 30 fps (sensor mode 1) |
| No NVMM interop in codebase | No NvBufSurface, no EGL, no cudaVideoDecoder — all GPU kernels use host RGB I/O |
| NVDEC present | `/dev/nvdec0` exists; `nvv4l2decoder` GStreamer element available |
| VIC (Video Image Compositor) | Hardware block used by nvvidconv — essentially free for resize/convert |

---

## Options Evaluated

### Option D — Software upscale (sws_scale before writeBmp)
**Discarded.** Interpolated pixels, no sensor data. Defeated the purpose.

### Option A — Two concurrent Argus sessions on same sensor
**Not viable.** libargus creates one `ICaptureSession` per CSI port. The Tegra ISP/VI
produces one output resolution per sensor at a time. Attempting a second
`nvarguscamerasrc sensor-id=0` while one is active fails at the Argus daemon level.
The CONTEXT.md note "Orin Nano supports 2 concurrent Argus sessions" means two
*different* sensors, not two sessions on the same sensor.

### Option B — Encode at 4K, downscale for transport
**Not viable at 4056×3040.** x264enc at full res ≈ 89% CPU — starves YOLO.
**Viable at 2028×1520** (~28% CPU) as a minimal-change fallback if everything else
fails: 3 MP stills, 30 fps stream, encode branch change only.

### Option C — V4L2 direct still capture (bypass Argus)
**Not viable for IMX477 on Jetson.** IMX477 is a MIPI CSI-2 sensor. `/dev/video0` is
part of the Tegra media graph owned exclusively by `nvargus-daemon` when a session is
active. The existing `v4l2_backend.cpp` is designed for USB webcams (MJPEG pipeline)
and is unrelated to CSI cameras. The original recommendation of Option C was based
on incorrect assumptions about CSI camera V4L2 accessibility.

### Chosen Approach — Tee pipeline with GPU-space processing
Run Argus at 4056×3040@15fps, use a GStreamer tee to split into:
- **still_sink branch**: full-res NV12, pull-on-demand for BMP saves
- **encode branch**: nvvidconv hardware downscale → x264enc at 720p (same CPU cost as before)

In the future (Phases 2–3), move all GPU processing (filters, YOLO resize) into
the NV12 NVMM domain, eliminating the current encode→decode→RGB round-trip.

---

## Current Data Flow (After Phase 1)

```
IMX477 sensor (Jetson Orin Nano)
  |
  v
NvidiaArgusBackend::Start(sensor_id, encode_w, encode_h, fps)
  |  src/cpp_accelerator/adapters/camera/backends/nvidia_argus_backend.cpp
  |  Argus source ALWAYS at 4056×3040@15fps (kSensorWidth/Height/Fps)
  |  encode_w/h default: 1280×720 (kDefaultEncodeWidth/Height)
  |
  v
GStreamer tee pipeline:
  |
  +-> [still_sink branch]
  |     queue leaky=2 max-size-buffers=1
  |     ! nvvidconv (VIC HW: NVMM pitch-aligned → system-mem stride=width)
  |     ! video/x-raw,width=4056,height=3040,format=NV12
  |     ! appsink name=still_sink  ← pull-on-demand, max-buffers=1 drop=true
  |
  +-> [encode branch]
        nvvidconv (VIC HW: 4K → 720p)
        ! video/x-raw,width=1280,height=720,format=I420
        ! x264enc tune=zerolatency speed-preset=ultrafast ...
        ! h264parse ! video/x-h264,stream-format=byte-stream,alignment=au
        ! appsink name=proc_sink  ← continuous H.264 callbacks

  v
GstCameraSource → CameraHub::Subscribe  ← fans out H.264 AUs (unchanged)
  |  src/cpp_accelerator/adapters/camera/gst_camera_source.h
  |  src/cpp_accelerator/adapters/camera/camera_hub.h
  v
BirdWatcher::OnH264Frame(data, info)     ← receives H.264 AU (unchanged)
  |  bird_watcher.cpp
  v
FeedDecoderAndExtractRgb()               ← libavcodec H.264 → RGB at 720p (unchanged)
  v
Two consumers, SAME 720p rgb buffer:
  |
  +-> DetectBird(rgb, w, h)              ← YOLO inference via TensorRT (unchanged)
  |
  +-> MaybeSave → SaveCapture()          ← NOW pulls 4K NV12 from still_sink
        camera_hub_->GrabStillFrame(sensor_id, &w, &h)
        → NV12→RGB24 via sws_scale (CPU, one-shot, ~50ms for 12MP)
        → writeBmp at 4056×3040 (12.3 MP)

GrabStillFrame delegation chain:
  CameraHub::GrabStillFrame(sensor_id, out_w, out_h)
    → GstCameraSource::GrabStillFrame
      → GstCameraSourceImpl::GrabStillFrame
        → NvidiaArgusBackend::GrabStillFrame
          → gst_app_sink_try_pull_sample(still_sink, 500ms timeout)
          → gst_buffer_map → rtc::binary (NV12, stride=width)
```

---

## Phase 1 — COMPLETE

**Goal:** 12.3 MP BMP stills with no stream interruption. No kernel changes.

### Files Modified

| File | Change |
|------|--------|
| `adapters/camera/backends/camera_backend.h` | Added `virtual GrabStillFrame(out_w, out_h)` (default returns `{}`) |
| `adapters/camera/backends/nvidia_argus_backend.h` | Added `GrabStillFrame` override declaration |
| `adapters/camera/backends/nvidia_argus_backend.cpp` | **Full restructure**: tee pipeline, `still_sink`+`proc_sink`, `GrabStillFrame` impl, `DetectCameras` advertises both modes |
| `adapters/camera/gst_camera_source_impl.h/.cpp` | Added `GrabStillFrame` delegation |
| `adapters/camera/gst_camera_source.h/.cpp` | Added `GrabStillFrame` delegation |
| `adapters/camera/camera_hub.h/.cpp` | Added `GrabStillFrame(sensor_id, out_w, out_h)` |
| `application/bird_watch/bird_watcher.cpp` | `SaveCapture` rewrote to pull 4K NV12 + convert to RGB |

### Key Constants (nvidia_argus_backend.cpp)

```cpp
constexpr int kSensorWidth  = 4056;
constexpr int kSensorHeight = 3040;
constexpr int kSensorFps    = 15;
constexpr int kDefaultEncodeWidth  = 1280;
constexpr int kDefaultEncodeHeight = 720;
```

### Verification Command (run on Jetson)

```bash
# Validate tee pipeline reaches PLAYING before deploying C++ binary
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=4056,height=3040,framerate=15/1,format=NV12' ! \
  tee name=t \
    t. ! queue leaky=2 max-size-buffers=1 ! \
       nvvidconv ! 'video/x-raw,width=4056,height=3040,format=NV12' ! fakesink \
    t. ! nvvidconv ! 'video/x-raw,width=1280,height=720,format=I420' ! \
       x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 \
               intra-refresh=true key-int-max=60 ! \
       'video/x-h264,profile=baseline' ! fakesink -e

# After C++ binary runs with a bird event, verify BMP size:
file captures/*.bmp    # must report 4056 x 3040
```

---

## Phase 2 — GPU-Space Filter + YOLO Pipeline (NOT STARTED)

**Goal:** Eliminate the encode→decode→RGB round-trip for the camera path. Keep all
processing in GPU memory (NVMM/CUDA) from Argus through YOLO and the WebRTC encoder.

### Current Bottleneck (what Phase 2 removes)

```
proc_sink (H.264 bytes)
  → BirdWatcher::OnH264Frame (queue)
  → libavcodec avcodec_send_packet / avcodec_receive_frame  [CPU decode]
  → sws_scale YUV→RGB24  [CPU]
  → RGB24 host memory
  → cudaMemcpy H→D  [host→device copy]
  → CUDA gray/blur kernel  [GPU]
  → cudaMemcpy D→H  [device→host copy]
  → RGB24 host memory
  → cuda_letterbox_resize_to_device (H→D copy again)  [GPU]
  → TensorRT YOLO
  → re-encode to H.264 (CPU)  [WebRTC path, live_video_processor.cpp]
```

### Target (Phase 2)

```
proc_sink (720p NV12, continuous)
  → NvBufSurfaceMap → CUDA device ptr
  → [CUDA: NV12→RGBA in-place]
  → [CUDA: gray/blur device-ptr kernels]
  → [CUDA: RGBA→NV12 in-place]
  → push NV12 into encode_appsrc → x264enc → H.264 → WebRTC
  ↘ [CUDA: NV12→float32 letterbox 640×640] → TensorRT YOLO
```

### New Files to Create

| File | Purpose |
|------|---------|
| `adapters/camera/nvbuf_cuda_utils.h/.cpp` | `MapNvmmBuffer(GstBuffer*)` → `NvmmFrame{y_ptr, uv_ptr, width, height, pitch}` using `NvBufSurfaceFromFd` + `NvBufSurfaceMap` |
| `adapters/compute/cuda/kernels/nv12_utils_kernel.h/.cu` | `cuda_nv12_to_rgba_device`, `cuda_rgba_to_nv12_device`, `cuda_nv12_letterbox_device` (replaces `cuda_letterbox_resize_to_device`) |
| `adapters/compute/cuda/gpu_frame_processor.h/.cpp` | Orchestrates the GPU-space pipeline per 720p NV12 frame |
| `adapters/camera/encode_pipeline.h/.cpp` | `appsrc → nvvidconv → x264enc → appsink` pipeline for re-encoding processed frames |

### Files to Modify in Phase 2

| File | Change |
|------|--------|
| `adapters/compute/cuda/kernels/grayscale_kernel.cu` | Add `cuda_convert_to_grayscale_device(uint8_t* rgba_dev, ...)` (device-ptr overload, no H→D/D→H) |
| `adapters/compute/cuda/kernels/blur/separable_basic.cu` | Add `cuda_apply_gaussian_blur_separable_device(uint8_t* rgba_dev, ...)` |
| `adapters/compute/cuda/kernels/letterbox_kernel.cu/.h` | Remove (replaced by `cuda_nv12_letterbox_device`) |
| `application/bird_watch/bird_watcher.cpp/.h` | Remove libavcodec decode loop, AVCodecContext/AVFrame/SwsContext members; subscribe to `proc_sink` NV12 callback; wire `GpuFrameProcessor` |

### Dependency to Add (Dockerfile)

```dockerfile
# docker-cpp-dependencies/Dockerfile
RUN apt-get install -y libnvbuf-utils-dev

# docker-cuda-runtime/Dockerfile
RUN apt-get install -y libnvbuf-utils
```

### Proc_sink Pipeline Change Needed for Phase 2

The current Phase 1 `proc_sink` branch outputs **H.264 bytes** (encode branch). For Phase 2,
`proc_sink` must output **720p NV12 in NVMM** instead. The GStreamer pipeline for the
encode branch must change from:

```
# Phase 1 (current)
cam_tee. ! nvvidconv ! video/x-raw,width=1280,height=720,format=I420 !
         x264enc ... ! appsink name=proc_sink
```

to:

```
# Phase 2
cam_tee. ! nvvidconv !
         video/x-raw(memory:NVMM),width=1280,height=720,format=NV12 !
         appsink name=proc_sink emit-signals=true max-buffers=2 drop=true
```

Encoding is then handled by the new `EncodePipeline` class (appsrc → x264enc → appsink)
which receives GPU-processed NV12 frames pushed by `GpuFrameProcessor`.

### Key Risk

`NvBufSurfaceMap` requires the `libnvbuf-utils` package. Confirm availability:
```bash
ls /usr/lib/aarch64-linux-gnu/libnvbuf_utils.so  # must exist in container
```

---

## Phase 3 — WebRTC Inbound Hardware Decode (NOT STARTED)

**Goal:** Inbound H.264 from browser peers also stays in GPU space through filters.

### Current Bottleneck

`live_video_processor.cpp` decodes with libavcodec software decoder, converts via
sws_scale on CPU, applies CUDA filters (H→D/D→H per filter), re-encodes via libavcodec.

### Target

```
H.264 bytes from peer
  → appsrc → h264parse → nvv4l2decoder (NVDEC HW)
  → NVMM NV12
  → same GpuFrameProcessor as Phase 2
  → EncodePipeline (appsrc → x264enc → H.264)
  → WebRTC track
```

### Platform Detection

```cpp
// nvv4l2decoder available on Orin Nano (NVDEC present, unlike NVENC)
const bool has_nvdec = HasGstElement("nvv4l2decoder") && (access("/dev/nvdec0", F_OK) == 0);
// Fall back to existing libavcodec path on x86 dev machine
```

### File to Modify

`adapters/webrtc/live_video_processor.cpp` — add an optional GStreamer decode pipeline
(`appsrc → h264parse → nvv4l2decoder → appsink`) selected at construction; reuse
`GpuFrameProcessor` from Phase 2 for the filter + encode path.

---

## Existing Kernel Inventory (for Phase 2 reference)

| File | Function | Input | Output | Notes |
|------|----------|-------|--------|-------|
| `kernels/letterbox_kernel.cu` | `cuda_letterbox_resize_to_device` | `uint8_t* host RGB` | `float* device` normalized | **Remove in Phase 2** |
| `kernels/grayscale_kernel.cu` | `cuda_convert_to_grayscale` | `uint8_t* host RGB` | `uint8_t* host grayscale` | Add device-ptr overload |
| `kernels/blur/separable_basic.cu` | `cuda_apply_gaussian_blur_separable` | `uint8_t* host RGB` | `uint8_t* host RGB` | Add device-ptr overload |
| `kernels/blur/separable_tiled.cu` | `cuda_apply_gaussian_blur_separable` (tiled) | `uint8_t* host RGB` | `uint8_t* host RGB` | Shared memory, best perf |

All current kernels use host memory I/O: they allocate temp device buffers internally,
do `cudaMemcpy H→D`, run kernel, `cudaMemcpy D→H`. Phase 2 adds device-ptr overloads
that skip both copies.

---

## GStreamer Pipeline Reference (current, after Phase 1)

```
nvarguscamerasrc sensor-id=0 wbmode=4 !
video/x-raw(memory:NVMM),width=4056,height=3040,framerate=15/1,format=NV12 !
tee name=cam_tee

cam_tee. ! queue leaky=2 max-size-buffers=1 !
         nvvidconv !
         video/x-raw,width=4056,height=3040,format=NV12 !
         appsink name=still_sink emit-signals=false sync=false max-buffers=1 drop=true

cam_tee. ! nvvidconv !
         video/x-raw,width=1280,height=720,format=I420 !
         x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000
                 vbv-buf-capacity=400 intra-refresh=true key-int-max=60 !
         video/x-h264,profile=baseline !
         h264parse config-interval=-1 !
         video/x-h264,stream-format=byte-stream,alignment=au !
         appsink name=proc_sink emit-signals=true max-buffers=2 drop=true
```

---

##  Git State

Phase 1 changes are uncommitted as of writing. Commit Phase 1 before starting Phase 2.

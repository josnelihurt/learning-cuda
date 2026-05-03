# BirdWatcher High-Resolution Stills

## Intent

Store captures at the maximum sensor resolution (IMX477: up to 4056x3040, ~12.3 MP)
while keeping the streaming + inference pipeline at a transport-viable resolution
(currently 1920x1080 or 1280x720).

## Current Data Flow

```
IMX477 sensor (Jetson Orin Nano)
  |
  v
NvidiaArgusBackend::Start(sensor_id, width, height, fps)    <-- single resolution
  |  src/cpp_accelerator/adapters/camera/backends/nvidia_argus_backend.cpp:307
  |  Pipeline: nvarguscamerasrc -> NV12 -> [nvvidconv -> nvv4l2h264enc | x264enc] -> appsink
  |  Defaults: 1920x1080@30fps (kDefaultWidth/kDefaultHeight/kDefaultFps, line 311-313)
  |  DetectCameras advertises: 1920x1080@60fps (line 287-289)
  v
GstCameraSource                                                <-- backend selector
  |  src/cpp_accelerator/adapters/camera/gst_camera_source.h
  v
CameraHub::Subscribe(sensor_id, width, height, fps, cb)       <-- fans out H.264 AUs
  |  src/cpp_accelerator/adapters/camera/camera_hub.h:58
  |  One GstCameraSource per sensor_id, shared across subscribers
  v
BirdWatcher::OnH264Frame(data, info)                          <-- receives H.264 AU
  |  bird_watcher.cpp:97
  v
FeedDecoderAndExtractRgb()                                    <-- libavcodec H.264 -> RGB
  |  bird_watcher.cpp:242
  |  RgbFromDecodedFrame() at pipeline resolution (sws_scale, line 293)
  v
Two consumers, SAME rgb buffer at pipeline resolution:
  |
  +-> DetectBird(rgb, w, h)                                  <-- YOLO inference
  |     bird_watcher.cpp:345
  |     via ProcessorEngine -> TensorRT
  |
  +-> MaybeSave -> SaveCapture(rgb, w, h)                    <-- BMP write
        bird_watcher.cpp:393-429
        image_sink_->writeBmp(path, rgb.data(), width, height, 3)
        src/cpp_accelerator/domain/interfaces/image_sink.h (IImageSink)

Config defaults (BirdWatcherConfig, bird_watcher.h:39):
  capture_width=1280, capture_height=720, capture_fps=30
```

### Key Constraint

There is **one Argus session per sensor** producing one H.264 stream at one resolution.
The BMP is written from the decoded H.264 frame, so it can never exceed the encode resolution.

## IMX477 Sensor Capabilities

| Resolution   | Max FPS | Notes                          |
|-------------|---------|--------------------------------|
| 4056x3040   | ~15-20  | Full sensor readout (~12.3 MP) |
| 3840x2160   | 30      | 4K UHD                         |
| 1920x1080   | 60      | Current pipeline target        |

Orin Nano has no hardware NVENC -- uses x264enc software encode.

## Options

### Option A: Two independent Argus sessions

Run a second `nvarguscamerasrc` at full sensor resolution solely for still capture.
The streaming pipeline stays at 1920x1080@30.

- Requires a "still capture" API on CameraHub: open Argus, grab one frame, close.
- Orin Nano typically supports 2 concurrent Argus sessions.
- Touches: `CameraBackend` interface (new method), `NvidiaArgusBackend`, `CameraHub`,
  `BirdWatcher::SaveCapture`.

Pros: true full-res stills, completely decoupled from stream.
Cons: double ISP load/memory bandwidth, limited concurrent sessions.

### Option B: Upscale pipeline resolution, downscale for transport

Set Argus to 3840x2160@30, encode H.264 at full res, decode at full res for BMPs.
Add a downscale step before WebRTC transport.

- The BirdWatcher decode loop already handles any resolution the H.264 stream carries.
- Transport layer needs a downscale step (GStreamer videoscale, or libyuv in WebRTC path).
- x264enc encoding 4K on Orin Nano (no NVENC) will be very slow.

Pros: single pipeline, full-res stills, no second Argus session.
Cons: heavy software encode at 4K, transport layer changes needed.

### Option C: V4L2 direct still capture (bypass Argus)

Keep Argus at 1920x1080@30 for streaming + inference.
When `MaybeSave` triggers, open `/dev/video0` via V4L2 directly, request a single
frame at max resolution, write it, close.

- IMX477 on Jetson typically allows concurrent Argus + V4L2 access.
- Requires new V4L2 still-capture utility code.
- Touches: new adapter class, `BirdWatcher::SaveCapture`.

Pros: zero impact on streaming, true sensor resolution, lightest on resources.
Cons: Argus/V4L2 contention risk (driver-dependent), new V4L2 code.

### Option D: Software upscale before BMP write

Upscale the decoded RGB via sws_scale before calling writeBmp.

- Trivial change in `SaveCapture`.
- No additional detail -- just bigger file with interpolated pixels.

Pros: simplest, no pipeline changes.
Cons: fake resolution, no additional detail, defeats the purpose.

## Recommendation

**Option C** is the best fit: true sensor resolution for captures, no streaming
pipeline changes, lightest on Orin Nano resources. If V4L2/Argus contention
proves problematic on the IMX477 driver, fall back to **Option A** with
on-demand single-frame Argus sessions.

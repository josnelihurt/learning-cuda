# Jetson Camera Backends (Argus/V4L2) - End-to-End Guide

This document explains how camera detection and camera streaming work in this project, with focus on Jetson Nano Orin production deployment.

It captures the implementation details and operational fixes applied during debugging of:

- cameras detected but stream not starting
- GStreamer plugin/library missing errors in container
- Argus daemon socket/device visibility issues in Docker

---

## 1) Quick Mental Model

There are two independent phases:

1. **Detection phase** (`ListCameras`)
   - Probes available camera backends.
   - Returns `RemoteCameraInfo` list (sensor IDs, display names, modes).
2. **Streaming phase** (`StartCameraStream`)
   - Starts live capture pipeline for selected camera.
   - Encodes H.264 and pushes frames to WebRTC track.

A system can pass detection but still fail streaming (different pipeline, different dependencies).

---

## 2) Backend Architecture in This Codebase

Main files:

- `camera_backend.h` - common backend interface
- `nvidia_argus_backend.cpp` - Jetson CSI backend (`nvarguscamerasrc`)
- `v4l2_backend.cpp` - V4L2/USB backend
- `stub_backend.cpp` - fallback no-op backend

Build wiring:

- `src/cpp_accelerator/adapters/camera/backends/BUILD`
- compile-time flags:
  - `--config=nvidia-argus-camera`
  - `--config=v4l2-camera`

Jetson production build currently enables Argus through:

- `src/cpp_accelerator/Dockerfile.build`
- ARM path uses `CAMERA_FLAGS="--config=nvidia-argus-camera"`

---

## 3) Current Jetson Argus Stream Pipeline

`NvidiaArgusBackend::Start(...)` uses:

```text
nvarguscamerasrc sensor-id=<id> !
video/x-raw(memory:NVMM),width=<w>,height=<h>,framerate=<fps>/1,format=NV12 !
nvvidconv !
nvv4l2h264enc insert-sps-pps=true bitrate=2000000 !
[optional h264parse config-interval=-1 !]
appsink name=sink emit-signals=true max-buffers=2 drop=true
```

Important runtime behavior implemented:

- If request arrives as invalid `0x0@0fps`, backend now normalizes to defaults:
  - `1920x1080@30`
- `h264parse` is now optional at runtime:
  - if present, used
  - if missing, pipeline still built without parser (with warning log)

---

## 4) Required Docker Runtime Dependencies (Jetson)

Installed in `src/cpp_accelerator/docker-cuda-runtime/Dockerfile`:

- `gstreamer1.0-tools`
- `gstreamer1.0-plugins-base`
- `gstreamer1.0-plugins-bad`
- `gstreamer1.0-plugins-good`
- `gstreamer1.0-plugins-ugly`
- `libgstreamer1.0-0`
- `libgstreamer-plugins-base1.0-0`
- `libegl1`
- `libgles2`
- `kmod`

Why each mattered in incidents:

- missing `libEGL.so.1` -> NVIDIA GStreamer plugins failed to load
- missing `libGLESv2.so.2` -> `nvarguscamerasrc` plugin load failures
- missing `h264parse` -> streaming pipeline parse error (`no element "h264parse"`)
- missing `kmod` tools -> noisy `lsmod/modprobe` warnings

Build-time dependencies (for compiling camera backends) live in:

- `src/cpp_accelerator/docker-cpp-dependencies/Dockerfile`
- includes:
  - `libgstreamer1.0-dev`
  - `libgstreamer-plugins-base1.0-dev`

---

## 5) Required Jetson Docker Mounts/Devices

`src/cpp_accelerator/docker-compose.yml` now maps required resources.

### Socket mount

- `/tmp/argus_socket:/tmp/argus_socket`

Without it, Argus daemon connection fails with:

- `Connecting to nvargus-daemon failed: No such file or directory`

### Device mappings

CSI + media graph + Tegra nodes:

- `/dev/video0`, `/dev/video1`
- `/dev/v4l-subdev0..3`
- `/dev/media0`
- `/dev/nvhost-ctrl-gpu`
- `/dev/nvhost-ctrl-isp`, `-isp-thi`
- `/dev/nvhost-ctrl-nvcsi`
- `/dev/nvhost-ctrl-vi0`, `-vi0-thi`
- `/dev/nvhost-ctrl-vi1`, `-vi1-thi`
- `/dev/nvhost-nvsched_ctrl_fifo-gpu`
- `/dev/nvmap`
- `/dev/nvgpu`

Without these, you can get:

- detection success but stream session creation failure
- `Failed to create CaptureSession`

---

## 6) Why Detection May Work But Streaming Fails

This happened repeatedly during debugging and is expected if runtime is incomplete.

Examples:

1. **Detection succeeded**:
   - `ProbeSensor(...)` with a minimal pipeline passed.
2. **Streaming failed**:
   - production pipeline had stricter caps/encoder/parser requirements
   - invalid request dimensions/fps (`0x0@0fps`)
   - missing element (`h264parse`)
   - missing mount/device for Argus session

Takeaway: always validate both phases independently.

---

## 7) Operational Diagnostics Checklist

Run on Jetson host:

```bash
ls -l /tmp/argus_socket
ls /dev/video* /dev/v4l-subdev* /dev/media*
```

Run inside container:

```bash
ls -l /tmp/argus_socket
gst-inspect-1.0 nvarguscamerasrc
gst-inspect-1.0 nvv4l2h264enc
gst-inspect-1.0 h264parse
```

Test minimal Argus probe:

```bash
gst-launch-1.0 -q nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink
```

Test stream pipeline shape:

```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12' ! \
  nvvidconv ! nvv4l2h264enc insert-sps-pps=true bitrate=2000000 ! \
  h264parse config-interval=-1 ! fakesink -e
```

Application-side logs to watch:

- `/tmp/cppaccelerator.log` (inside accelerator container)
- `cuda-go-server` logs for:
  - `ListCameras` camera count
  - session lifecycle

---

## 8) Debug Timeline of Real Failures and Fixes

1. Missing `/dev/nvhost-ctrl` style mappings from compose
   - fixed by mapping actual device set present on target Jetson

2. `libgstreamer-1.0.so.0` missing
   - fixed by runtime package install

3. `libEGL.so.1` missing
   - fixed by `libegl1`

4. `libGLESv2.so.2` missing
   - fixed by `libgles2`

5. Argus daemon socket unavailable
   - fixed with `/tmp/argus_socket` bind mount

6. Argus capture session failures due to missing media/subdev nodes
   - fixed with `v4l-subdev*`, `media0`, plus additional Tegra node mappings

7. `StartCameraStream` received `0x0@0fps`
   - fixed with parameter normalization in Argus backend

8. `h264parse` element absent in runtime image
   - fixed by adding `gstreamer1.0-plugins-bad`
   - backend also made resilient if parser is absent

---

## 9) Zero-to-Expert Learning Path (Jetson + GStreamer)

### Level 0 - Basic Concepts

- GStreamer pipeline = graph of elements connected with `!`
- source (`nvarguscamerasrc`) -> transform (`nvvidconv`) -> encoder (`nvv4l2h264enc`) -> sink (`appsink`)
- each element may come from a different plugin package

### Level 1 - Jetson Camera Basics

- CSI cameras on Jetson typically use Argus stack (`nvarguscamerasrc`)
- Argus requires daemon communication (`/tmp/argus_socket`)
- Docker isolation means you must pass socket and required `/dev/*` nodes explicitly

### Level 2 - Packaging and Runtime Contracts

- compile-time success does not guarantee runtime success
- verify both:
  - headers/libs at build time
  - runtime `.so` and plugin elements in final image
- use `gst-inspect-1.0 <element>` as first-line validation

### Level 3 - Production Reliability

- log backend selection and backend-specific failures
- normalize invalid client parameters to known-safe defaults
- prefer explicit, actionable logs over generic "failed to start stream"
- keep compose device list close to validated target hardware

### Level 4 - Performance/Quality Tuning

- tune `nvv4l2h264enc` (bitrate, profile, rate control, IDR interval)
- evaluate parser necessity for downstream packetizer contract
- profile frame flow and packetization behavior in WebRTC path

---

## 10) Authoritative External References

### NVIDIA Jetson Accelerated GStreamer

- NVIDIA Jetson Linux Developer Guide - Accelerated GStreamer  
  https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html

This doc includes:

- Jetson accelerated plugin usage (`nvarguscamerasrc`, `nvv4l2h264enc`)
- installation guidance (including `nvidia-l4t-gstreamer`)
- example pipelines with `h264parse`

### GStreamer Plugin Taxonomy and h264parse

- GStreamer "bad" plugins module page  
  https://gstreamer.freedesktop.org/modules/gst-plugins-bad.html
- h264parse plugin documentation  
  https://gstreamer.freedesktop.org/documentation/videoparsersbad/h264parse.html
- gst-plugins-bad source repository  
  https://github.com/GStreamer/gst-plugins-bad

Note: freedesktop pages may block automated fetch in some environments, but those are the official upstream docs.

### Community Jetson-in-Docker Argus Context

- NVIDIA Developer Forums (Argus in Docker discussion)  
  https://forums.developer.nvidia.com/t/using-nvarguscamerasrc-within-a-docker-container/328246

Treat forum threads as practical guidance, not canonical specification.

---

## 11) Practical Runbook (Production Jetson)

1. Deploy latest compose and image.
2. Verify socket/devices mounted in container.
3. Verify required elements:
   - `nvarguscamerasrc`
   - `nvv4l2h264enc`
   - `h264parse` (or confirm backend fallback warning if absent)
4. Trigger `ListCameras`:
   - expect `camera_count > 0`
5. Trigger `StartCameraStream`:
   - ensure logs do not show `0x0@0fps` unhandled
   - ensure Argus pipeline enters PLAYING
6. If stream still black:
   - inspect WebRTC stats/channel logs
   - verify encoded frames are emitted from appsink callback

---

## 12) Key Rule for This Project

For Jetson production camera path, keep Argus explicit and fail loudly on real backend issues.

Do not hide Argus failures behind compile-time backend fallbacks that change operational behavior silently.

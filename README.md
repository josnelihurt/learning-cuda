# CUDA Image Processor

Playing around with CUDA and real-time video processing. Mostly just wanted to see how fast I could get grayscale filters running on GPU vs CPU.

![Screenshot](./data/screenshot.png)

## What's this?

Webcam app that processes video through CUDA kernels. You pick filters, it runs them on your GPU (or CPU if you want to compare), shows you the FPS. That's it.

Built with Go because I wanted to try CGO, and C++ for the actual processing. Probably overengineered but whatever, it works.

## Setup

Need HTTPS for camera access (browser thing):
```bash
./scripts/setup-ssl.sh
./scripts/start-dev.sh --build
```

Then go to `https://localhost:8443`

## Tech

- Go server handling WebSocket
- C++/CUDA doing the processing
- CGO gluing them together
- Bazel because... honestly not sure why I picked Bazel but it's fine
- Caddy for HTTPS (needed for getUserMedia API)

Frontend is just vanilla JS. Didn't want to deal with React or whatever.

## Grayscale algorithms

Implemented a few different ones:
- **BT.601** (0.299R + 0.587G + 0.114B) - old TV standard
- **BT.709** (0.2126R + 0.7152G + 0.0722B) - HD standard  
- **Average** - simple (R+G+B)/3
- **Lightness** - (max+min)/2
- **Luminosity** - weighted average, similar to BT.601

Can switch between them in the UI. Honestly can't tell much difference except Average looks a bit off.

## GPU vs CPU

The GPU version is way faster obviously. At 720p:
- GPU: ~150 FPS
- CPU: ~25 FPS

CPU implementation is still useful for debugging though.

## Commands

```bash
./scripts/start-dev.sh          # start (uses existing build)
./scripts/start-dev.sh --build  # rebuild everything first
./scripts/kill-services.sh      # kill all processes
```

Frontend hot reloads. For C++/Go you gotta rebuild.

## Code structure

```
cpp_accelerator/
  infrastructure/cuda/  - GPU kernels
  infrastructure/cpu/   - CPU versions
  ports/cgo/           - CGO bridge

webserver/
  cmd/server/          - main.go
  web/                 - static files

scripts/               - bash stuff
```

## How it works

1. Browser grabs webcam frames
2. Sends via WebSocket as base64 PNG
3. Go decodes, passes to C++ via CGO
4. CUDA kernel processes on GPU
5. Result goes back as base64
6. Browser renders it

Stats bar shows FPS and processing time per frame. Logs every 30 frames.

## Filters

Right now only grayscale but the pipeline supports chaining filters. You can drag and drop to reorder them. Might add blur or edge detection later if I feel like it.

## Issues

- Sometimes Caddy doesn't stop cleanly, have to `pkill` it
- Camera permission on Linux needs your user in `video` group
- SSL cert warnings if you didn't run setup script
- Toast notifications are kinda big but I like them

## Todo maybe

- [ ] Blur filter
- [ ] Edge detection (Sobel?)
- [ ] Maybe try OpenCL instead of CUDA
- [ ] Better error handling when GPU isn't available
- [ ] Docker setup

## Notes

This started as a weekend project to learn CUDA and turned into... this. Code quality is mixed - some parts are clean, others not so much. Feel free to use it if you want but no guarantees.

The CGO part was actually harder than the CUDA part tbh.

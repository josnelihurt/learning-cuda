# CUDA Image Processor

Playing around with CUDA and real-time video processing. Mostly just wanted to see how fast I could get grayscale filters running on GPU vs CPU.

![Screenshot](./data/screenshot.png)

## What's this?

Webcam app that processes video through CUDA kernels. You pick filters, it runs them on your GPU (or CPU if you want to compare), shows you the FPS. That's it.

Built with Go because I wanted to try CGO, and C++ for the actual processing. Probably overengineered but whatever, it works.

## Setup

### Development Mode

```bash
./scripts/setup-ssl.sh
./scripts/start-dev.sh --build  # Dev mode (hot reload)
# or
./scripts/start-dev.sh --build --prod  # Production bundle
```

### Docker Deployment

Production deployment with GPU acceleration using Docker Compose and Traefik:

```bash
# Validate environment (checks SSL certs, Docker, NVIDIA Container Toolkit, GPU)
./scripts/validate-docker-env.sh

# Build and run
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f app

# Stop containers
docker-compose down
```

Access the application:
- **Application**: https://localhost
- **Traefik Dashboard**: http://localhost:8081

Requirements:
- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU with drivers
- SSL certificates in `.secrets/` directory (run `./scripts/setup-ssl.sh`)

The Docker setup uses:
- Multi-stage build (frontend → backend → runtime)
- NVIDIA CUDA 12.5 runtime
- Traefik for HTTPS termination
- Full GPU passthrough to container

## Tech

- Go server handling WebSocket
- C++/CUDA doing the processing
- CGO gluing them together
- Bazel because... honestly not sure why I picked Bazel but it's fine
- Caddy for HTTPS (needed for getUserMedia API)

Frontend: Lit Web Components + TypeScript with Vite bundler. No React, just native web components.

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

## Known issues

- Caddy sometimes doesn't stop cleanly, use `pkill caddy`
- Camera needs user in `video` group on Linux
- SSL cert warnings if you skip setup script

## Roadmap

- Blur filter (Gaussian)
- Edge detection (Sobel)
- Try OpenCL as alternative to CUDA
- Better GPU unavailable handling
- ~~Docker container~~ ✓

## Notes

Weekend project that grew. Code quality varies. CGO was harder than CUDA.

# CUDA Image Processor Web Server

Modern web server for CUDA-accelerated image processing with clean architecture.

## Quick Start

### Run Directly with Bazel
```bash
bazel run //webserver/cmd/server:server
```

Then open http://localhost:8080 in your browser.

## Docker

### Build and Load Image into Docker
```bash
# Build the Docker image
bazel build //webserver:server_image

# Load image into Docker daemon
bazel run //webserver:server_load
```

### Run with Docker
```bash
# Run the container
docker run -p 8080:8080 cuda-webserver:latest

# Or run in background
docker run -d -p 8080:8080 cuda-webserver:latest
```

### Image Details
- **Base Image**: Google Distroless (Debian 12) - minimal, secure container
- **Size**: ~25MB (static binary + minimal runtime)
- **Includes**: 
  - Go web server binary
  - HTML templates
  - Sample image (lena.png)

## Development

### Add Go Dependencies
```bash
# Add package
go get <package>

# Update BUILD files automatically
bazel run //:gazelle

# Update MODULE.bazel
bazel mod tidy
```

### Build Everything
```bash
bazel build //...
```

## Architecture

```
webserver/
├── cmd/server/          # Main entry point
├── internal/
│   ├── domain/          # Business entities & interfaces
│   ├── application/     # Use cases
│   ├── infrastructure/  # External services (C++ connector stub)
│   └── interfaces/http/ # HTTP handlers & WebSocket
└── web/templates/       # HTML templates
```

## Features

- ✅ Modern, responsive UI
- ✅ WebSocket support for real-time updates
- ✅ Clean Architecture pattern
- ✅ Dockerized with Bazel
- 🔄 C++ CUDA integration (stub - coming soon)


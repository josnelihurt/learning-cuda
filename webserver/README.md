# CUDA Image Processor - Web Server

GPU-accelerated image processing web application using CUDA, Go, C++, and Protocol Buffers.

## Development Mode

For frontend development with **hot reload** (no recompilation needed):

### Quick Start

```bash
# From project root
./dev.sh
```

This will:
1. Build the server
2. Start in development mode
3. Watch for changes in HTML/CSS/JS
4. Reload templates on each request (F5 to see changes)

### Manual Command

```bash
# Build first
bazel build //webserver/cmd/server:server

# Run in dev mode from project root
./bazel-bin/webserver/cmd/server/server_/server \
    -dev \
    -webroot=/home/jrb/code/cuda-learning/webserver/web
```

### Frontend Files

Edit these files and just hit **F5** in your browser:
- `webserver/web/templates/index.html` - HTML structure
- `webserver/web/static/css/main.css` - Styles
- `webserver/web/static/js/app.js` - JavaScript logic

**No recompilation needed!** Changes are loaded on each request.

## Production Mode

For production (embedded files, single binary):

```bash
bazel run //webserver/cmd/server:server
```

Templates and static files are embedded in the binary.

## Architecture

```
webserver/
├── cmd/server/          # Main entry point
├── internal/
│   ├── application/     # Use cases
│   ├── domain/          # Business logic
│   ├── infrastructure/  # External integrations (CGO/C++)
│   └── interfaces/      # HTTP handlers
└── web/
    ├── templates/       # HTML templates
    └── static/
        ├── css/         # Stylesheets
        └── js/          # JavaScript
```

## Features

- **CUDA Acceleration**: GPU-powered image processing
- **CGO Integration**: Go ↔ C++ via Protocol Buffers
- **Hot Reload**: Dev mode for rapid frontend iteration
- **Clean Architecture**: Separation of concerns
- **WebSocket**: Real-time communication ready

## Filters Available

- **None**: Original image (no processing)
- **Grayscale**: CUDA kernel conversion (~267ms)

Future: Blur, Edge Detection, Custom filters...

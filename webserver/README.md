# CUDA Image Processor - Web Server

GPU-accelerated image processing web application using CUDA, Go, C++, and Protocol Buffers.

## Development Mode

For local development with hot reload:

### Quick Start

From project root:
```bash
./scripts/dev/start.sh --build  # First time or after code changes
./scripts/dev/start.sh           # Subsequent runs
```

This will:
1. Build the server and frontend
2. Start services (Flipt, Jaeger, etc.)
3. Enable hot reload for frontend (Vite dev server)
4. Start the Go server with hot reload support

**Access:** https://localhost:8443

### Manual Build

Build the Go server:
```bash
cd webserver
make build
```

Or from project root:
```bash
cd webserver && make build
```

Run the server:
```bash
cd webserver
make run
```

Or with config file:
```bash
./bin/server -config=config/config.yaml
```

### Development Mode

Run server with hot reload:
```bash
cd webserver
make dev
```

## Production Mode

For production (embedded files, single binary):

```bash
cd webserver
make build
./bin/server -config=config/config.production.yaml
```

The frontend is built with Vite and embedded as static assets. Templates and static files are served from the binary.

## Architecture

```
webserver/
├── cmd/server/          # Main entry point (main.go)
├── pkg/
│   ├── application/     # Use cases (business logic)
│   ├── domain/          # Domain models and interfaces
│   ├── infrastructure/  # External integrations
│   │   ├── processor/   # C++/CUDA plugin loader
│   │   ├── featureflags/# Flipt integration
│   │   └── build/       # Build info repository
│   ├── interfaces/      # HTTP/WebSocket/Connect-RPC handlers
│   │   ├── connectrpc/  # Connect-RPC handlers
│   │   ├── websocket/   # WebSocket handlers
│   │   └── statichttp/  # Static file serving
│   ├── config/          # Configuration management
│   ├── container/       # Dependency injection
│   └── telemetry/       # OpenTelemetry integration
└── web/
    ├── src/             # TypeScript source (Lit Web Components)
    ├── templates/       # HTML templates
    └── static/          # Static assets (CSS, images)
```

## Features

- **CUDA Acceleration**: GPU-powered image processing via dynamic plugin system (dlopen)
- **Connect-RPC**: Type-safe RPC with HTTP/JSON and gRPC support
- **Vanguard**: RESTful API transcoding using google.api.http annotations
- **Protocol Buffers**: Multiple proto services (config_service, file_service, image_processor_service)
- **Hot Reload**: Frontend development with Vite, Go hot reload for templates
- **Clean Architecture**: Domain → Application → Infrastructure → Interfaces layers
- **WebSocket**: Real-time video/image processing
- **OpenTelemetry**: Distributed tracing integration

## Protocol Buffers

The project uses multiple proto service definitions:

- `proto/config_service.proto` - Configuration and system info (with REST annotations)
- `proto/file_service.proto` - File upload and listing (with REST annotations)
- `proto/image_processor_service.proto` - Image processing operations (with REST annotations)
- `proto/common.proto` - Shared message types

All services include `google.api.http` annotations for RESTful routing via Vanguard transcoder.

Generate code:
```bash
./scripts/build/protos.sh
# Or manually:
docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
```

## Frontend

The frontend uses:
- **Lit Web Components** - Native web components
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server
- **Vitest** - Unit testing
- **Playwright** - E2E testing

**Development:**
```bash
cd webserver/web
npm install
npm run dev  # Vite dev server
```

**Build:**
```bash
npm run build
```

## See Also

- [Main README](../README.md) - Project overview and setup
- [Testing Documentation](../docs/testing-and-coverage.md) - Test execution guide

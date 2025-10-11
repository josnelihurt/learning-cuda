# Docker Quick Start

## Prerequisites Check

```bash
# 1. Validate your environment
./scripts/validate-docker-env.sh
```

This checks:
- ✓ SSL certificates exist in `.secrets/`
- ✓ Docker is installed and running
- ✓ NVIDIA Container Toolkit is configured
- ✓ GPU is accessible

## Run the Application

```bash
# Option 1: Using the helper script (recommended)
./scripts/docker-run.sh

# Option 2: Using docker-compose directly
docker-compose up --build
```

## Access Points

Once running:
- **Application**: https://localhost
- **Traefik Dashboard**: http://localhost:8081

## Common Commands

```bash
# Run in background
./scripts/docker-run.sh --detach

# View logs
docker-compose logs -f app

# Stop everything
docker-compose down

# Rebuild from scratch
docker-compose down
docker-compose up --build
```

## Troubleshooting

### GPU Not Working?
```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
```

### Port 443 Already in Use?
```bash
# Find what's using it
sudo lsof -i :443

# Stop Caddy if running from dev setup
./scripts/kill-services.sh
```

### SSL Certificate Warnings?
This is expected for localhost development with self-signed certificates. Click "Advanced" → "Proceed to localhost" in your browser.

## What Gets Built

1. **Frontend**: TypeScript/Lit components → Vite bundle
2. **Backend**: C++/CUDA + Go → Single binary with CGO
3. **Runtime**: Minimal CUDA runtime image (~2GB)

## Architecture

```
Traefik (Port 443, HTTPS)
    ↓
App Container (Port 8080)
    ↓
NVIDIA GPU (full passthrough)
```

## More Info

- Full documentation: [DOCKER.md](./DOCKER.md)
- Development setup: [README.md](./README.md)
- SSL setup: `./scripts/setup-ssl.sh`


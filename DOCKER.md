# Docker Deployment Guide

This guide explains how to deploy the CUDA Image Processor using Docker with GPU acceleration.

## Architecture

The Docker setup consists of three build stages:

1. **Frontend Builder**: Compiles TypeScript/Lit components with Vite
2. **Backend Builder**: Builds C++/CUDA code and Go server with Bazel
3. **Runtime**: Minimal CUDA runtime image with compiled artifacts

The deployment uses Docker Compose with two services:
- **app**: The CUDA Image Processor (with GPU passthrough)
- **traefik**: Reverse proxy handling HTTPS termination

## Prerequisites

### Required Software

1. **Docker** (with Docker Compose)
   ```bash
   # Check installation
   docker --version
   docker-compose --version
   ```

2. **NVIDIA Container Toolkit**
   ```bash
   # Install on Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **NVIDIA GPU with drivers**
   ```bash
   # Verify GPU is accessible
   nvidia-smi
   ```

### SSL Certificates

SSL certificates must exist in the `.secrets/` directory:

```bash
./scripts/setup-ssl.sh
```

This creates:
- `.secrets/localhost+2.pem` (certificate)
- `.secrets/localhost+2-key.pem` (private key)

## Quick Start

### 1. Validate Environment

Run the validation script to check all prerequisites:

```bash
./scripts/validate-docker-env.sh
```

This checks:
- SSL certificates exist
- Docker is installed and running
- NVIDIA Container Toolkit is configured
- GPU is accessible

### 2. Build and Run

#### Option A: Using the helper script

```bash
# Interactive mode (logs in terminal)
./scripts/docker-run.sh

# Detached mode (runs in background)
./scripts/docker-run.sh --detach
```

#### Option B: Using docker-compose directly

```bash
# Build and run (interactive)
docker-compose up --build

# Build and run (detached)
docker-compose up -d --build

# View logs
docker-compose logs -f app

# Stop containers
docker-compose down
```

### 3. Access the Application

- **Application**: https://localhost
- **Traefik Dashboard**: http://localhost:8081

## Configuration

### GPU Configuration

The `docker-compose.yml` configures GPU access:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu, compute, utility]
```

To limit GPU usage, modify the `count` field or specify specific GPU indices.

### Port Configuration

Default ports:
- **443**: HTTPS (Traefik)
- **8081**: Traefik Dashboard
- **8080**: Internal app port (not exposed)

To change ports, edit `docker-compose.yml`:

```yaml
traefik:
  ports:
    - "443:443"    # Change first number for host port
    - "8081:8080"  # Traefik dashboard
```

### SSL Certificates

Traefik loads certificates from `.secrets/` (mounted as `/certs` in container).

To use different certificates:
1. Place certificates in `.secrets/` directory
2. Update `traefik-config.yml`:
   ```yaml
   tls:
     certificates:
       - certFile: /certs/your-cert.pem
         keyFile: /certs/your-key.pem
   ```

## Troubleshooting

### GPU Not Available in Container

**Symptom**: Application fails to use GPU

**Solution**:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### SSL Certificate Errors

**Symptom**: Browser shows certificate warnings

**Solution**: The certificates are self-signed for localhost. This is expected for local development. Click "Advanced" and proceed to the site.

To avoid warnings:
- Install the certificate authority in your system's trust store
- Use valid certificates from Let's Encrypt or similar

### Build Fails in Bazel Stage

**Symptom**: Docker build fails during backend compilation

**Solutions**:
1. Check CUDA compatibility:
   ```bash
   nvcc --version  # Should be 12.5.x
   ```

2. Increase Docker memory:
   - Docker Desktop: Settings → Resources → Memory (set to 8GB+)
   - Linux: No limit by default

3. Clean build cache:
   ```bash
   docker-compose down
   docker system prune -a
   docker-compose up --build
   ```

### Port Already in Use

**Symptom**: `Error: port 443 already in use`

**Solution**:
```bash
# Find process using port
sudo lsof -i :443

# Stop conflicting service
sudo systemctl stop nginx  # or apache2, caddy, etc.
```

### Traefik Not Starting

**Symptom**: Traefik container exits immediately

**Check logs**:
```bash
docker-compose logs traefik
```

**Common issues**:
- Certificate files not found: Verify `.secrets/` directory exists
- Invalid traefik-config.yml: Check YAML syntax
- Docker socket permission: Add user to docker group

## Development vs Production

### Development Setup

For active development with hot reload:
```bash
./scripts/start-dev.sh --build
```

This runs:
- Vite dev server (hot reload for frontend)
- Go server with `-dev` flag
- Caddy for HTTPS

### Production Setup

For production deployment:
```bash
./scripts/docker-run.sh --detach
```

This uses:
- Pre-compiled frontend assets
- Optimized runtime image
- Traefik for HTTPS
- No dev tools or hot reload

## Advanced Usage

### Custom Bazel Build Flags

Edit `Dockerfile` to add custom Bazel flags:

```dockerfile
RUN bazel build \
    --copt=-O3 \
    --cxxopt=-march=native \
    //webserver/cmd/server:server \
    //cpp_accelerator/ports/cgo:cgo_api
```

### Multiple GPU Nodes

To deploy across multiple GPU nodes:

1. Use Docker Swarm or Kubernetes
2. Configure GPU scheduling based on workload
3. Use persistent volumes for shared data

Example Kubernetes manifest would need NVIDIA device plugin.

### Monitoring

Add Prometheus/Grafana for monitoring:

```yaml
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## File Reference

### Created Files

- **Dockerfile**: Multi-stage build configuration
- **docker-compose.yml**: Service orchestration with GPU
- **traefik-config.yml**: Traefik TLS and routing configuration
- **.dockerignore**: Files excluded from Docker context
- **scripts/validate-docker-env.sh**: Pre-deployment validation
- **scripts/docker-run.sh**: Convenient wrapper for docker-compose

### Important Directories

- `.secrets/`: SSL certificates (mounted read-only)
- `webserver/web/`: Frontend source and compiled assets
- `cpp_accelerator/`: C++/CUDA processing code
- `bazel-bin/`: Build artifacts (excluded from Docker context)

## Security Considerations

1. **Certificate Storage**: Keep `.secrets/` secure, never commit to git
2. **Traefik Dashboard**: Port 8081 exposed for debugging, disable in production
3. **Docker Socket**: Mounted read-only to Traefik, minimize exposure
4. **Root User**: Runtime container uses root, consider adding non-root user for production

## Performance Tuning

### Build Performance

- Use BuildKit for faster builds:
  ```bash
  DOCKER_BUILDKIT=1 docker-compose build
  ```

- Cache Bazel artifacts between builds (add volume):
  ```yaml
  app:
    volumes:
      - bazel-cache:/root/.cache/bazel
  ```

### Runtime Performance

- GPU memory: Monitor with `nvidia-smi` inside container
- CPU cores: Limit with `cpus` in docker-compose.yml
- Network: Use host networking for minimal latency (removes isolation)

## Support

For issues specific to:
- **Docker setup**: Check this guide and validation script
- **CUDA errors**: Verify NVIDIA drivers and Container Toolkit
- **Application bugs**: See main README.md

## License

Same as main project.


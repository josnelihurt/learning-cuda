#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Validating Docker environment..."
echo ""

ERRORS=0

echo "[1/5] Checking SSL certificates..."
if [ ! -f "$PROJECT_ROOT/.secrets/localhost+2.pem" ]; then
    echo "  ERROR: SSL certificate not found"
    echo "  Run: ./scripts/docker/generate-certs.sh"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: Certificate found"
fi

if [ ! -f "$PROJECT_ROOT/.secrets/localhost+2-key.pem" ]; then
    echo "  ERROR: SSL key not found"
    echo "  Run: ./scripts/docker/generate-certs.sh"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: Certificate key found"
fi
echo ""

echo "[2/5] Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "  ERROR: Docker is not installed"
    echo "  Install: https://docs.docker.com/get-docker/"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: Docker is installed"
fi
echo ""

echo "[3/5] Checking Docker daemon..."
if ! docker info &> /dev/null; then
    echo "  ERROR: Docker daemon is not running"
    echo "  Start: sudo systemctl start docker"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: Docker daemon is running"
fi
echo ""

echo "[4/5] Checking NVIDIA Container Toolkit..."
if ! docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "  ERROR: NVIDIA Container Toolkit not configured"
    echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: NVIDIA Container Toolkit configured"
fi
echo ""

echo "[5/5] Checking GPU availability..."
if ! nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not found or GPU not available"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: GPU available"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
fi
echo ""

if [ $ERRORS -eq 0 ]; then
    echo "All validation checks passed"
    echo ""
    echo "Build and run: docker compose up --build"
    echo "Access: https://localhost"
    echo "Dashboard: http://localhost:8081"
    echo ""
    exit 0
else
    echo "Validation failed with $ERRORS error(s)"
    echo "Fix errors above before building"
    exit 1
fi

# Optional: Validate Cloudflare Tunnel configuration
# Call this function from scripts that need Cloudflare validation
validate_cloudflare() {
    echo "[6/6] Checking Cloudflare Tunnel configuration..."
    
    if [ ! -f "$PROJECT_ROOT/.secrets/cloudflare.env" ]; then
        echo "  ERROR: Cloudflare env file not found: .secrets/cloudflare.env"
        return 1
    fi
    
    source "$PROJECT_ROOT/.secrets/cloudflare.env"
    
    if [ -z "$TUNNEL_TOKEN" ]; then
        echo "  ERROR: TUNNEL_TOKEN not set in .secrets/cloudflare.env"
        return 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/cloudflared-config.yml" ]; then
        echo "  ERROR: cloudflared-config.yml not found"
        return 1
    fi
    
    echo "  OK: Cloudflare Tunnel configured"
    echo ""
    return 0
}

# Export function so it can be called from other scripts
export -f validate_cloudflare 2>/dev/null || true

#!/bin/bash
# Build Docker images for production using buildx
# This script builds natively for the current platform (ARM64 on Jetson)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "${BLUE}Building Docker images for production (native platform)${NC}"
echo "Project root: ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

# Detect current architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    PLATFORM="linux/arm64"
    echo -e "${GREEN}Detected ARM64 architecture (Jetson Nano)${NC}"
elif [ "$ARCH" = "x86_64" ]; then
    PLATFORM="linux/amd64"
    echo -e "${GREEN}Detected AMD64 architecture${NC}"
else
    echo -e "${RED}Unsupported architecture: $ARCH${NC}"
    exit 1
fi

# Create buildx builder if it doesn't exist
if ! docker buildx inspect cuda-builder &>/dev/null; then
    echo -e "${YELLOW}Creating buildx builder 'cuda-builder'${NC}"
    docker buildx create --name cuda-builder --driver docker-container --use
else
    echo -e "${GREEN}Using existing buildx builder 'cuda-builder'${NC}"
    docker buildx use cuda-builder
fi

# Build for current platform
echo -e "${YELLOW}Building images for platform: ${PLATFORM}${NC}"
docker buildx build \
    --platform "${PLATFORM}" \
    --load \
    -f Dockerfile \
    -t cuda-learning-app:latest \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  - Run: docker compose --profile production up -d"
    echo "  - Or use: ./deployment/scripts/deploy.sh --production"
else
    echo -e "${RED}Build failed. Check the output above for errors.${NC}"
    exit 1
fi

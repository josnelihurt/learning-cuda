#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Go Linter (Docker)..."
cd "$PROJECT_ROOT"

# Create a temporary stub for the loader package to avoid CGO issues
LOADER_DIR="webserver/pkg/infrastructure/processor/loader"
TEMP_DIR="/tmp/cuda-learning-loader-backup"
if [ -d "$LOADER_DIR" ]; then
    mv "$LOADER_DIR" "$TEMP_DIR"
    mkdir -p "$LOADER_DIR"
    # Create a minimal stub file
    cat > "$LOADER_DIR/loader_stub.go" << 'EOF'
package loader

import (
    "fmt"
    pb "github.com/jrb/cuda-learning/proto/gen"
)

const CurrentAPIVersion = "2.0.0"

type Loader struct{
    Path string
}

type Registry struct{}

func NewLoader(libraryPath string) (*Loader, error) {
    return nil, fmt.Errorf("loader not available during linting")
}

func NewRegistry(libraryBasePath string) *Registry {
    return &Registry{}
}

func (r *Registry) ListVersions() []string {
    return []string{}
}

func (r *Registry) GetAllLibraries() map[string]int {
    return map[string]int{}
}

func (r *Registry) LoadLibrary(version string) (*Loader, error) {
    return nil, fmt.Errorf("loader not available during linting")
}

func (r *Registry) Discover() error {
    return fmt.Errorf("loader not available during linting")
}

func (r *Registry) GetByVersion(version string) (*Loader, error) {
    return nil, fmt.Errorf("loader not available during linting")
}

func (l *Loader) Init(req *pb.InitRequest) (*pb.InitResponse, error) {
    return nil, fmt.Errorf("loader not available during linting")
}

func (l *Loader) ProcessImage(req *pb.ProcessImageRequest) (*pb.ProcessImageResponse, error) {
    return nil, fmt.Errorf("loader not available during linting")
}

func (l *Loader) GetCapabilities(req *pb.GetCapabilitiesRequest) (*pb.GetCapabilitiesResponse, error) {
    return nil, fmt.Errorf("loader not available during linting")
}

func (l *Loader) Cleanup() {}

func (l *Loader) CachedCapabilities() *pb.LibraryCapabilities {
    return nil
}

func (l *Loader) GetVersion() string {
    return "stub"
}

func (l *Loader) IsCompatibleWith(apiVersion string) bool {
    return false
}
EOF
fi

# Run the linter
docker compose -f docker-compose.dev.yml --profile lint run --rm lint-golang || {
    # Restore the directory even if linting fails
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$LOADER_DIR"
        mv "$TEMP_DIR" "$LOADER_DIR"
    fi
    echo "FAILED: Go linter"
    exit 1
}

# Restore the directory
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$LOADER_DIR"
    mv "$TEMP_DIR" "$LOADER_DIR"
fi

echo "Go linter passed" // emoji-allowed


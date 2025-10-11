#!/bin/bash
# Run the application with HTTPS support
# This is a simple launcher - run in two separate terminals

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if certificates exist
if [ ! -f ".secrets/localhost+2.pem" ] || [ ! -f ".secrets/localhost+2-key.pem" ]; then
    echo "⚠️  SSL certificates not found!"
    echo ""
    echo "Generate them first:"
    echo "  ./scripts/setup-ssl.sh"
    echo ""
    exit 1
fi

echo "🚀 CUDA Image Processor - HTTPS Setup"
echo ""
echo "Start these in SEPARATE terminals:"
echo ""
echo "Terminal 1 - Go Server:"
echo "  cd $PROJECT_ROOT"
echo "  bazel-bin/webserver/cmd/server/server_/server"
echo ""
echo "Terminal 2 - Caddy HTTPS Proxy:"
echo "  cd $PROJECT_ROOT"
echo "  caddy run"
echo ""
echo "Then open: https://localhost:8443"
echo ""


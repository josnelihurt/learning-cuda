#!/bin/bash
# Start development environment with HTTPS support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if certificates exist
if [ ! -f ".secrets/localhost+2.pem" ] || [ ! -f ".secrets/localhost+2-key.pem" ]; then
    echo "⚠️  SSL certificates not found!"
    echo ""
    echo "Run setup first:"
    echo "  ./scripts/setup-ssl.sh"
    echo ""
    exit 1
fi

# Check if Caddy is installed
if ! command -v caddy &> /dev/null; then
    echo "⚠️  Caddy is not installed!"
    echo ""
    echo "Install Caddy:"
    echo "  Ubuntu/Debian:"
    echo "    sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https"
    echo "    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg"
    echo "    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list"
    echo "    sudo apt update"
    echo "    sudo apt install caddy"
    echo ""
    echo "  Or download from: https://caddyserver.com/download"
    echo ""
    exit 1
fi

echo "🚀 Starting CUDA Image Processor Development Environment"
echo ""
echo "Starting services:"
echo "  - Go Server on http://localhost:8080"
echo "  - Caddy HTTPS Proxy on https://localhost:443"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $GO_PID $CADDY_PID 2>/dev/null
    wait $GO_PID $CADDY_PID 2>/dev/null
    echo "✅ Services stopped"
}

trap cleanup EXIT INT TERM

# Start Go server in background
echo "▶️  Starting Go server..."
bazel-bin/webserver/cmd/server/server_/server &
GO_PID=$!
sleep 2

# Check if Go server started successfully
if ! kill -0 $GO_PID 2>/dev/null; then
    echo "❌ Failed to start Go server"
    exit 1
fi

# Start Caddy in background
echo "▶️  Starting Caddy HTTPS proxy..."
caddy run --config Caddyfile 2>&1 | sed 's/^/[Caddy] /' &
CADDY_PID=$!
sleep 2

# Check if Caddy started successfully
if ! kill -0 $CADDY_PID 2>/dev/null; then
    echo "❌ Failed to start Caddy"
    kill $GO_PID 2>/dev/null
    exit 1
fi

echo ""
echo "✅ All services running!"
echo ""
echo "📍 URLs:"
echo "  HTTPS: https://localhost"
echo "  HTTP:  http://localhost (redirects to HTTPS)"
echo "  Direct: http://localhost:8080 (Go server)"
echo ""
echo "📝 Logs:"
echo "  Caddy: ./caddy.log"
echo ""
echo "Press Ctrl+C to stop all services..."
echo ""

# Wait for processes
wait $GO_PID $CADDY_PID


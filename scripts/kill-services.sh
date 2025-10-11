#!/bin/bash
# Kill all services (Go server, Caddy)

set -e

echo "Stopping services..."

# Kill Go server
pkill -f "server_/server" 2>/dev/null && echo "Go server stopped" || echo "Go server not running"

# Kill Caddy
if pgrep -x "caddy" > /dev/null; then
    pkill -x caddy 2>/dev/null && echo "Caddy stopped" || {
        echo "Warning: Failed to kill Caddy. Try: sudo pkill caddy"
        exit 1
    }
else
    echo "Caddy not running"
fi

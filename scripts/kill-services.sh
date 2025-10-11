#!/bin/bash
# Kill all services (Go server, Caddy, etc.)

echo "🛑 Killing all services..."

# Kill Go server
pkill -f "server_/server" && echo "✓ Go server killed" || echo "✗ No Go server running"

# Kill Caddy (may need sudo for system instance)
if pgrep -x "caddy" > /dev/null; then
    echo "Found Caddy process(es):"
    ps aux | grep caddy | grep -v grep
    echo ""
    echo "Attempting to kill Caddy..."
    pkill -x caddy && echo "✓ Caddy killed" || {
        echo "⚠️  Failed to kill Caddy. You may need sudo:"
        echo "  sudo pkill caddy"
    }
else
    echo "✗ No Caddy running"
fi

echo ""
echo "✅ Done!"


#!/bin/bash
set -e

echo "Stopping services..."

pkill -f "server_/server" 2>/dev/null && echo "Go server stopped" || echo "Go server not running"

pgrep -x "caddy" > /dev/null && {
    pkill -x caddy 2>/dev/null && echo "Caddy stopped" || echo "Caddy not running"
}

pkill -f "vite" 2>/dev/null && echo "Vite stopped" || echo "Vite not running"

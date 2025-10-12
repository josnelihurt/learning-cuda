#!/bin/bash

echo "Stopping services..."

pkill -f "server_/server" 2>/dev/null && echo "Go server stopped" || echo "Go server not running"

pkill -x caddy 2>/dev/null && echo "Caddy stopped" || echo "Caddy not running"
sleep 0.5
pkill -9 -x caddy 2>/dev/null || true

pkill -f "vite" 2>/dev/null && echo "Vite stopped" || echo "Vite not running"

fuser -k 2019/tcp 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
fuser -k 8443/tcp 2>/dev/null || true

sleep 1
echo "Services stopped"

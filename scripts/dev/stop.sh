#!/bin/bash

echo "Stopping services..."

pkill -f "server_/server" 2>/dev/null && echo "Go server stopped" || echo "Go server not running"

pkill -f "vite" 2>/dev/null && echo "Vite stopped" || echo "Vite not running"

fuser -k 2019/tcp 2>/dev/null || true
fuser -k 8443/tcp 2>/dev/null || true

sleep 1
echo "Services stopped"

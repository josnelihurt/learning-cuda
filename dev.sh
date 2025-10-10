#!/bin/bash
# Development mode script for CUDA Image Processor
# This script runs the server in dev mode with hot reload for frontend files

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_ROOT="${PROJECT_ROOT}/webserver/web"

echo "🚀 Starting server in DEVELOPMENT mode"
echo "📁 Web root: ${WEB_ROOT}"
echo ""
echo "✨ You can now edit HTML/CSS/JS files and refresh your browser (F5)"
echo "   No need to recompile!"
echo ""
echo "🌐 Open: http://localhost:8080"
echo ""

bazel run //webserver/cmd/server:server -- -dev -webroot="${WEB_ROOT}"


#!/bin/bash
# Start Cloudflare tunnel for CUDA processor

echo "🌐 Starting Cloudflare Tunnel..."
echo ""
echo "URLs available:"
echo "  • https://webtest-cuda.josnelihurt.me (permanent)"
echo ""

cloudflared tunnel run cuda-processor


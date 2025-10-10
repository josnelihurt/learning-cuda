#!/bin/bash
# Setup SSL certificates for local HTTPS development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SECRETS_DIR="$PROJECT_ROOT/.secrets"

echo "🔐 Setting up SSL certificates for local development..."
echo ""

# Check if mkcert is installed
if ! command -v mkcert &> /dev/null; then
    echo "⚠️  mkcert is not installed."
    echo ""
    echo "Would you like to install it? (recommended)"
    echo "  1) Install mkcert (Ubuntu/Debian)"
    echo "  2) Use OpenSSL (self-signed, browser warnings)"
    echo "  3) Exit"
    read -p "Choose option [1-3]: " choice
    
    case $choice in
        1)
            echo "📦 Installing mkcert..."
            if [ ! -f /usr/local/bin/mkcert ]; then
                wget -O /tmp/mkcert https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-amd64
                chmod +x /tmp/mkcert
                sudo mv /tmp/mkcert /usr/local/bin/
                sudo apt-get install -y libnss3-tools 2>/dev/null || true
            fi
            ;;
        2)
            echo "📝 Using OpenSSL for self-signed certificate..."
            cd "$SECRETS_DIR"
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout localhost+2-key.pem \
                -out localhost+2.pem \
                -subj "/C=US/ST=State/L=City/O=Dev/CN=localhost"
            echo "✅ Self-signed certificate created!"
            echo "⚠️  Note: Your browser will show security warnings for self-signed certs."
            exit 0
            ;;
        3)
            echo "❌ Exiting..."
            exit 0
            ;;
        *)
            echo "❌ Invalid option"
            exit 1
            ;;
    esac
fi

# Install mkcert CA
echo "📋 Installing local Certificate Authority..."
mkcert -install

# Generate certificates
echo "🔑 Generating SSL certificates..."
cd "$SECRETS_DIR"
mkcert localhost 127.0.0.1 ::1

echo ""
echo "✅ SSL certificates generated successfully!"
echo ""
echo "📁 Certificates location: $SECRETS_DIR/"
ls -lh "$SECRETS_DIR"/*.pem 2>/dev/null || echo "No .pem files found"
echo ""
echo "🚀 Next steps:"
echo "  1. Start your Go server: bazel run //webserver/cmd/server:server"
echo "  2. Start Caddy: caddy run"
echo "  3. Open browser: https://localhost"
echo ""


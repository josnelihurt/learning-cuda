#!/bin/bash
# Setup SSL certificates for local HTTPS development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SECRETS_DIR="$PROJECT_ROOT/.secrets"

echo "Setting up SSL certificates for local development"
echo ""

# Check if mkcert is installed
if ! command -v mkcert &> /dev/null; then
    echo "Error: mkcert not found"
    echo ""
    read -p "Install mkcert? [Y/n]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Installing mkcert..."
        wget -q -O /tmp/mkcert https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-amd64
        chmod +x /tmp/mkcert
        sudo mv /tmp/mkcert /usr/local/bin/
        sudo apt-get install -y libnss3-tools 2>/dev/null || true
        echo "mkcert installed"
    else
        echo "Aborted"
        exit 1
    fi
fi

# Install mkcert CA
echo "Installing local Certificate Authority..."
mkcert -install

# Get local IP and hostname
LOCAL_IP=$(hostname -I | awk '{print $1}')
HOSTNAME=$(hostname -s)

echo "Detected: $LOCAL_IP, $HOSTNAME"
echo ""

# Generate certificates
echo "Generating certificates..."
cd "$SECRETS_DIR"
mkcert localhost 127.0.0.1 ::1 $LOCAL_IP $HOSTNAME $HOSTNAME.local

echo ""
echo "Done. Certificates in: $SECRETS_DIR/"
echo "Start dev server with: ./scripts/start-dev.sh"
echo ""

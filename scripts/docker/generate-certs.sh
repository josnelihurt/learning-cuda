#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

CERTS_DIR=".secrets"
mkdir -p "$CERTS_DIR"

echo "Generating SSL certificates for local development..."

docker run --rm \
    -v "$PROJECT_ROOT/$CERTS_DIR:/certs" \
    alpine:3.19 \
    sh -c "
        apk add --no-cache openssl > /dev/null 2>&1 &&
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout /certs/localhost+2-key.pem \
            -out /certs/localhost+2.pem \
            -subj '/C=US/ST=Dev/L=Local/O=Development/CN=localhost' \
            -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1' &&
        chmod 644 /certs/localhost+2.pem /certs/localhost+2-key.pem &&
        echo 'Certificates generated successfully'
    "

echo ""
echo "SSL certificates created in $CERTS_DIR/"
echo "  Certificate: localhost+2.pem"
echo "  Private key: localhost+2-key.pem"
echo ""
echo "Note: Browser will show security warning (self-signed cert)"
echo "      This is normal for local development"


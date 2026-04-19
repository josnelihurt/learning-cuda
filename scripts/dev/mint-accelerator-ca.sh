#!/usr/bin/env bash
# mint-accelerator-ca.sh — creates the dev CA for the accelerator mTLS trust chain.
# Tool choice: OpenSSL (universally available, no extra dependency).
#
# Run once per dev machine. The CA cert is distributed to both the cloud Go server
# (as the client CA bundle) and the C++ accelerator (as the trust root for the server cert).
# NEVER commit accelerator-ca-key.pem.

set -euo pipefail

SECRETS_DIR="${SECRETS_DIR:-.secrets}"
mkdir -p "$SECRETS_DIR"

if [[ -f "$SECRETS_DIR/accelerator-ca.pem" ]]; then
  echo "CA already exists at $SECRETS_DIR/accelerator-ca.pem; refusing to overwrite."
  echo "Delete it (and accelerator-ca-key.pem) manually if you want to regenerate."
  exit 1
fi

echo "Generating dev CA private key..."
openssl genrsa -out "$SECRETS_DIR/accelerator-ca-key.pem" 4096

echo "Generating self-signed dev CA certificate (10 years)..."
openssl req -x509 -new -nodes \
  -key "$SECRETS_DIR/accelerator-ca-key.pem" \
  -sha256 -days 3650 \
  -subj "/CN=cuda-learning accelerator dev CA/O=cuda-learning dev" \
  -out "$SECRETS_DIR/accelerator-ca.pem"

echo ""
echo "Dev CA created:"
echo "  $SECRETS_DIR/accelerator-ca.pem       (trust root — share with all peers)"
echo "  $SECRETS_DIR/accelerator-ca-key.pem   (private key — NEVER commit or share)"
echo ""
echo "Next steps:"
echo "  ./scripts/dev/mint-accelerator-cert.sh --name accelerator-server --type server"
echo "  ./scripts/dev/mint-accelerator-cert.sh --name dev-accelerator-client --type client"

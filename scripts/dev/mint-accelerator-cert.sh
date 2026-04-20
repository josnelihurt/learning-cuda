#!/usr/bin/env bash
# mint-accelerator-cert.sh — generate a server or client cert signed by the dev CA.
#
# Usage:
#   ./scripts/dev/mint-accelerator-cert.sh --name accelerator-server --type server
#   ./scripts/dev/mint-accelerator-cert.sh --name dev-accelerator-client --type client
#
# Requires: openssl, mint-accelerator-ca.sh to have been run first.

set -euo pipefail

NAME=""
TYPE=""
SECRETS_DIR="${SECRETS_DIR:-.secrets}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --type) TYPE="$2"; shift 2 ;;
    --secrets-dir) SECRETS_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; echo "Usage: --name <cn> --type server|client [--secrets-dir <dir>]"; exit 1 ;;
  esac
done

if [[ -z "$NAME" || -z "$TYPE" ]]; then
  echo "Usage: --name <cn> --type server|client"
  exit 1
fi

if [[ "$TYPE" != "server" && "$TYPE" != "client" ]]; then
  echo "Error: --type must be 'server' or 'client'"
  exit 1
fi

CA_CERT="$SECRETS_DIR/accelerator-ca.pem"
CA_KEY="$SECRETS_DIR/accelerator-ca-key.pem"

if [[ ! -f "$CA_CERT" || ! -f "$CA_KEY" ]]; then
  echo "Error: CA not found. Run ./scripts/dev/mint-accelerator-ca.sh first."
  exit 1
fi

KEY="$SECRETS_DIR/$NAME-key.pem"
CERT="$SECRETS_DIR/$NAME.pem"
CSR="$SECRETS_DIR/$NAME.csr"
EXT_FILE="$SECRETS_DIR/$NAME.ext"

echo "Generating private key for $NAME..."
openssl genrsa -out "$KEY" 4096

echo "Generating CSR..."
openssl req -new -key "$KEY" -subj "/CN=$NAME/O=cuda-learning dev" -out "$CSR"

echo "Generating extensions file for type=$TYPE..."
if [[ "$TYPE" == "server" ]]; then
  cat > "$EXT_FILE" <<EOF
subjectAltName = DNS:localhost, IP:127.0.0.1, DNS:$NAME, DNS:*.josnelihurt.me
extendedKeyUsage = serverAuth
keyUsage = digitalSignature, keyEncipherment
EOF
else
  cat > "$EXT_FILE" <<EOF
extendedKeyUsage = clientAuth
keyUsage = digitalSignature
EOF
fi

echo "Signing certificate with dev CA (1 year)..."
openssl x509 -req \
  -in "$CSR" \
  -CA "$CA_CERT" \
  -CAkey "$CA_KEY" \
  -CAcreateserial \
  -out "$CERT" \
  -days 365 \
  -sha256 \
  -extfile "$EXT_FILE"

rm -f "$CSR" "$EXT_FILE"

echo ""
echo "Certificate written:"
echo "  $CERT"
echo "  $KEY"
echo ""
echo "Verify with:"
echo "  openssl verify -CAfile $CA_CERT $CERT"

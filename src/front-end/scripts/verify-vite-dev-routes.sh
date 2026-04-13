#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--inner" ]]; then
  exec timeout 45s "$0" --inner
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SECRETS_DIR="$(cd "$FRONTEND_DIR/.." && pwd)/.secrets"

if [[ ! -f "$SECRETS_DIR/localhost+2-key.pem" || ! -f "$SECRETS_DIR/localhost+2.pem" ]]; then
  echo "verify-vite-dev-routes: missing TLS files under .secrets/ (localhost+2-key.pem, localhost+2.pem)" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${VITE_PID:-}" ]] && kill -0 "$VITE_PID" 2>/dev/null; then
    kill "$VITE_PID" 2>/dev/null || true
    wait "$VITE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "$FRONTEND_DIR"
npm run dev -- --host 127.0.0.1 &
VITE_PID=$!

ready=0
for _ in $(seq 1 90); do
  if curl -skf --connect-timeout 1 "https://127.0.0.1:3000/react" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done

if [[ "$ready" -ne 1 ]]; then
  echo "verify-vite-dev-routes: timed out waiting for https://127.0.0.1:3000/react" >&2
  exit 1
fi

react_body=$(curl -skf --connect-timeout 2 "https://127.0.0.1:3000/react")
lit_body=$(curl -skf --connect-timeout 2 "https://127.0.0.1:3000/lit")

echo "$react_body" | rg -q "static/css/main.css" || {
  echo "verify-vite-dev-routes: /react missing static/css/main.css" >&2
  exit 1
}
echo "$react_body" | rg -q 'id="root"' || {
  echo "verify-vite-dev-routes: /react missing #root" >&2
  exit 1
}
echo "$react_body" | rg -q "/src/react/main\\.tsx" || {
  echo "verify-vite-dev-routes: /react missing React entry script" >&2
  exit 1
}

echo "$lit_body" | rg -q "static/css/main.css" || {
  echo "verify-vite-dev-routes: /lit missing static/css/main.css" >&2
  exit 1
}
echo "$lit_body" | rg -q "app-root" || {
  echo "verify-vite-dev-routes: /lit missing app-root" >&2
  exit 1
}

echo "verify-vite-dev-routes: OK (127.0.0.1:3000)"

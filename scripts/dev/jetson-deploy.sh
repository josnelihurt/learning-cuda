#!/usr/bin/env bash
# Run on the Jetson host, from /opt/josnelihurt/cuda-learning/
#
# Usage:
#   ./deploy.sh                         # reuse last image + wbmode from .env
#   ./deploy.sh 4.1.17                  # bump only the version (proto+arch fixed)
#   ./deploy.sh 4.1.17 5                # version + new wbmode
#   WBMODE=5 ./deploy.sh                # only change wbmode
#
# Persists ACCELERATOR_IMAGE and NVARGUS_WBMODE in ./.env so that subsequent
# `docker compose` invocations pick them up without re-exporting.

set -euo pipefail

REGISTRY='ghcr.io/josnelihurt-code/learning-cuda/cpp-accelerator'
PROTO_TAG='proto4.2.0'
ARCH='arm64'

cd "$(dirname "$0")"
ENV_FILE='./.env'
touch "$ENV_FILE"

# Read existing values from .env so unspecified args don't get clobbered.
read_env() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- || true; }
write_env() {
  local key="$1" val="$2"
  if grep -qE "^$key=" "$ENV_FILE"; then
    sed -i "s|^$key=.*|$key=$val|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$val" >> "$ENV_FILE"
  fi
}

CURRENT_IMAGE="$(read_env ACCELERATOR_IMAGE)"
CURRENT_WBMODE="$(read_env NVARGUS_WBMODE)"

# Resolve VERSION argument: positional $1, or extract from previous image, or fail.
VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  if [[ -n "$CURRENT_IMAGE" ]]; then
    VERSION="$(echo "$CURRENT_IMAGE" | sed -nE 's|.*:cpp-accelerator-([0-9.]+)-.*|\1|p')"
  fi
fi
if [[ -z "$VERSION" ]]; then
  echo "error: no version given and none found in $ENV_FILE" >&2
  echo "usage: $0 <X.Y.Z> [wbmode]" >&2
  exit 1
fi

# Resolve WBMODE: positional $2, env override, current value, or default 4.
WBMODE="${2:-${WBMODE:-${CURRENT_WBMODE:-4}}}"

IMAGE="${REGISTRY}:cpp-accelerator-${VERSION}-${PROTO_TAG}-${ARCH}"

write_env ACCELERATOR_IMAGE "$IMAGE"
write_env NVARGUS_WBMODE "$WBMODE"

echo "=== Deploy ==="
echo "  image:  $IMAGE"
echo "  wbmode: $WBMODE"
echo

docker compose pull
docker compose up -d --remove-orphans

echo
echo "=== Last 20 log lines ==="
sleep 2
docker logs --tail 20 cuda-accelerator-client 2>&1 || true

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REGISTRY="${REGISTRY:-ghcr.io}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt-code/learning-cuda}"
ARCH="${ARCH:-arm64}"
IMAGE_BASE="${REGISTRY}/${BASE_IMAGE_PREFIX}"

proto_latest="${IMAGE_BASE}/intermediate:proto-generated-latest-${ARCH}"
deps_ver="$(tr -d '[:space:]' < "${ROOT}/src/cpp_accelerator/docker-cpp-dependencies/VERSION")"
deps_img="${IMAGE_BASE}/intermediate:cpp-dependencies-${deps_ver}-${ARCH}"

pull_if_missing() {
  local ref="$1"
  if docker image inspect "${ref}" >/dev/null 2>&1; then
    echo "pull-ghcr-cpp-intermediates: present ${ref}"
    return 0
  fi
  echo "pull-ghcr-cpp-intermediates: pulling ${ref}"
  docker pull "${ref}"
}

if [[ "${PULL_PROTO_LATEST:-0}" == "1" ]]; then
  pull_if_missing "${proto_latest}"
fi

if [[ "${PULL_CPP_DEPENDENCIES:-0}" == "1" ]]; then
  pull_if_missing "${deps_img}"
fi

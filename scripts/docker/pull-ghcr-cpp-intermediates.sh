#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REGISTRY="${REGISTRY:-ghcr.io}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt-code/learning-cuda}"
ARCH="${ARCH:-arm64}"
IMAGE_BASE="${REGISTRY}/${BASE_IMAGE_PREFIX}"

proto_ver="$(tr -d '[:space:]' < "${ROOT}/proto/VERSION")"
proto_versioned="${IMAGE_BASE}/intermediate:proto-generated-${proto_ver}-${ARCH}"
proto_latest="${IMAGE_BASE}/intermediate:proto-generated-latest-${ARCH}"

deps_ver="$(tr -d '[:space:]' < "${ROOT}/src/cpp_accelerator/docker-cpp-dependencies/VERSION")"
deps_versioned="${IMAGE_BASE}/intermediate:cpp-dependencies-${deps_ver}-${ARCH}"
deps_latest="${IMAGE_BASE}/intermediate:cpp-dependencies-latest-${ARCH}"

cuda_runtime_ver="$(tr -d '[:space:]' < "${ROOT}/src/cpp_accelerator/docker-cuda-runtime/VERSION")"
cuda_runtime_versioned="${IMAGE_BASE}/base:cuda-runtime-${cuda_runtime_ver}-${ARCH}"
cuda_runtime_latest="${IMAGE_BASE}/base:cuda-runtime-latest-${ARCH}"

pull_if_missing() {
  local ref="$1"
  if docker image inspect "${ref}" >/dev/null 2>&1; then
    echo "pull-ghcr-cpp-intermediates: present ${ref}"
    return 0
  fi
  echo "pull-ghcr-cpp-intermediates: pulling ${ref}"
  docker pull "${ref}"
}

# cpp-builder reads the latest tag; cpp-accelerator reads the versioned one.
# Both point to the same digest on GHCR, but Docker treats them as separate
# local refs — pull the versioned one and locally retag as latest so every
# downstream stage finds its input.
ensure_local_alias() {
  local from="$1" to="$2"
  if docker image inspect "${to}" >/dev/null 2>&1; then return 0; fi
  docker tag "${from}" "${to}"
  echo "pull-ghcr-cpp-intermediates: tagged ${to} from ${from}"
}

if [[ "${PULL_PROTO_LATEST:-0}" == "1" ]]; then
  pull_if_missing "${proto_versioned}"
  ensure_local_alias "${proto_versioned}" "${proto_latest}"
fi

if [[ "${PULL_CPP_DEPENDENCIES:-0}" == "1" ]]; then
  pull_if_missing "${deps_versioned}"
  ensure_local_alias "${deps_versioned}" "${deps_latest}"
fi

if [[ "${PULL_CUDA_RUNTIME:-0}" == "1" ]]; then
  pull_if_missing "${cuda_runtime_versioned}"
  ensure_local_alias "${cuda_runtime_versioned}" "${cuda_runtime_latest}"
fi

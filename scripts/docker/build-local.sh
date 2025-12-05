#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REGISTRY="${REGISTRY:-local}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt/learning-cuda}"
ARCH_DEFAULT="$(uname -m)"
case "${ARCH_DEFAULT}" in
  x86_64) ARCH_DEFAULT="amd64" ;;
  aarch64) ARCH_DEFAULT="arm64" ;;
esac
ARCH="${ARCH_DEFAULT}"
REQUESTED_STAGES=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build local Docker images sequentially using docker build.

Options:
  --arch <arch>      Target architecture (amd64 or arm64). Defaults to host arch.
  --registry <name>  Registry prefix for tags. Defaults to value of \$REGISTRY or "local".
  --base-prefix <p>  Image namespace following the registry. Defaults to "josnelihurt/learning-cuda".
  --stage <name>     Build only the specified stage (can be passed multiple times).
  --list-stages      Print available stages and exit.
  -h, --help         Show this message.

Stages execute in the order defined internally. Without --stage the script runs all stages.
EOF
}

list_stages() {
  printf '%s\n' "${ALL_STAGES[@]}"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' not found" >&2
    exit 1
  fi
}

read_version() {
  local path="$1"
  if [[ ! -f "${REPO_ROOT}/${path}" ]]; then
    echo "Version file '${path}' not found" >&2
    exit 1
  fi
  tr -d '[:space:]' < "${REPO_ROOT}/${path}"
}

ALL_STAGES=(proto-tools go-builder bazel-base runtime-base integration-base proto cpp golang app grpc-server)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --base-prefix)
      BASE_IMAGE_PREFIX="$2"
      shift 2
      ;;
    --stage)
      REQUESTED_STAGES+=("$2")
      shift 2
      ;;
    --list-stages)
      list_stages
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${ARCH}" != "amd64" && "${ARCH}" != "arm64" ]]; then
  echo "Unsupported architecture '${ARCH}'. Use amd64 or arm64." >&2
  exit 1
fi

if [[ ${#REQUESTED_STAGES[@]} -eq 0 ]]; then
  REQUESTED_STAGES=("${ALL_STAGES[@]}")
else
  # Preserve declared order while intersecting with requested values.
  declare -A requested_map=()
  for stage in "${REQUESTED_STAGES[@]}"; do
    requested_map["$stage"]=1
  done
  filtered=()
  for stage in "${ALL_STAGES[@]}"; do
    if [[ -n "${requested_map[$stage]:-}" ]]; then
      filtered+=("$stage")
    fi
  done
  if [[ ${#filtered[@]} -eq 0 ]]; then
    echo "No valid stages requested. Available stages:" >&2
    list_stages >&2
    exit 1
  fi
  REQUESTED_STAGES=("${filtered[@]}")
fi

require_command docker

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not available" >&2
  exit 1
fi

if [[ -z "${BAZEL_REMOTE_CACHE:-}" ]]; then
  CACHE_HOST="192.168.10.80"
  CACHE_STATUS_URL="http://${CACHE_HOST}:9090/status"
  if curl -sf --connect-timeout 2 --max-time 4 "${CACHE_STATUS_URL}" >/dev/null 2>&1; then
    export BAZEL_REMOTE_CACHE="grpc://${CACHE_HOST}:9092"
    export BAZEL_REMOTE_UPLOAD_LOCAL_RESULTS="true"
    echo "Detected bazel-remote cache at ${CACHE_HOST}"
  else
    echo "Bazel-remote cache not reachable; proceeding without remote cache."
  fi
fi

HOST_ARCH="${ARCH_DEFAULT}"
if [[ "${ARCH}" != "${HOST_ARCH}" ]]; then
  echo "Warning: building for ${ARCH} on host ${HOST_ARCH} without buildx may fail." >&2
fi

cd "${REPO_ROOT}"

IMAGE_BASE="${REGISTRY}/${BASE_IMAGE_PREFIX}"
TARGETARCH="${ARCH}"

print_stage_header() {
  local label="$1"
  echo ""
  echo "=========================================="
  echo ">>> ${label}"
  echo "=========================================="
  echo ""
}

build_and_tag() {
  local tag="$1"
  local latest_tag="$2"
  local dockerfile="$3"
  local should_push="${4:-false}"
  shift 4
  local build_args=("$@")

  print_stage_header "docker build -f ${dockerfile}"
  echo "Tag (versioned): ${tag}"
  echo "Tag (latest):    ${latest_tag}"
  echo ""

  local docker_build_args=()
  
  if [[ "${REGISTRY}" != "local" ]]; then
    docker_build_args+=("--pull")
  fi
  
  local filtered_build_args=()
  for arg in "${build_args[@]}"; do
    if [[ "${arg}" == "--no-cache" ]]; then
      docker_build_args+=("--no-cache")
    else
      filtered_build_args+=("${arg}")
    fi
  done
  build_args=("${filtered_build_args[@]}")
  
  docker build \
    "${docker_build_args[@]}" \
    --build-arg "TARGETARCH=${TARGETARCH}" \
    "${build_args[@]}" \
    -f "${dockerfile}" \
    -t "${tag}" \
    -t "${latest_tag}" \
    "${REPO_ROOT}"

  docker image inspect "${tag}" >/dev/null 2>&1

  if [[ "${should_push}" == "true" && "${REGISTRY}" != "local" ]]; then
    echo "Pushing ${tag}..."
    docker push "${tag}" || true
    echo "Pushing ${latest_tag}..."
    docker push "${latest_tag}" || true
  fi
}

run_proto_tools() {
  local version
  version="$(read_version "proto/docker-build-base/VERSION")"
  local version_tag="${IMAGE_BASE}/base:proto-tools-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/base:proto-tools-latest-${ARCH}"

  print_stage_header "Building proto tools base (${version})"
  build_and_tag "${version_tag}" "${latest_tag}" "proto/docker-build-base/Dockerfile" "true"
}

run_go_builder() {
  local version
  version="$(read_version "webserver/builder/VERSION")"
  local version_tag="${IMAGE_BASE}/base:go-builder-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/base:go-builder-latest-${ARCH}"

  print_stage_header "Building go builder base (${version})"
  build_and_tag "${version_tag}" "${latest_tag}" "webserver/builder/Dockerfile" "true"
}

run_bazel_base() {
  local version
  version="$(read_version "cpp_accelerator/docker-build-base/VERSION")"
  local version_tag="${IMAGE_BASE}/base:bazel-base-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/base:bazel-base-latest-${ARCH}"

  print_stage_header "Building bazel base (${version})"
  build_and_tag "${version_tag}" "${latest_tag}" "cpp_accelerator/docker-build-base/Dockerfile" "true"
}

run_runtime_base() {
  local version
  version="$(read_version "runtime/VERSION")"
  local version_tag="${IMAGE_BASE}/base:runtime-base-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/base:runtime-base-latest-${ARCH}"

  print_stage_header "Building runtime base (${version})"
  build_and_tag "${version_tag}" "${latest_tag}" "runtime/Dockerfile" "true"
}

run_integration_base() {
  local version
  version="$(read_version "integration/VERSION")"
  local version_tag="${IMAGE_BASE}/base:integration-tests-base-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/base:integration-tests-base-latest-${ARCH}"

  print_stage_header "Building integration base (${version})"
  build_and_tag "${version_tag}" "${latest_tag}" "integration/Dockerfile" "true"
}

run_proto_generated() {
  local version
  version="$(read_version "proto/VERSION")"
  local version_tag="${IMAGE_BASE}/intermediate:proto-generated-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/intermediate:proto-generated-latest-${ARCH}"

  print_stage_header "Building proto intermediate (${version})"
  build_and_tag \
    "${version_tag}" \
    "${latest_tag}" \
    "proto/Dockerfile" \
    "true" \
    "--build-arg" "BASE_REGISTRY=${IMAGE_BASE}" \
    "--build-arg" "BASE_TAG=latest" \
    "--build-arg" "PROTO_VERSION=${version}"
}

run_cpp_built() {
  local proto_version
  local cpp_version
  proto_version="$(read_version "proto/VERSION")"
  cpp_version="$(read_version "cpp_accelerator/VERSION")"
  local version_tag="${IMAGE_BASE}/intermediate:cpp-built-${cpp_version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/intermediate:cpp-built-latest-${ARCH}"
  
  local bazel_base_image="${IMAGE_BASE}/base:bazel-base-latest-${ARCH}"
  local proto_generated_image="${IMAGE_BASE}/intermediate:proto-generated-latest-${ARCH}"
  
  if ! docker image inspect "${bazel_base_image}" >/dev/null 2>&1; then
    echo "Error: Base image ${bazel_base_image} not found. Build bazel-base first." >&2
    exit 1
  fi
  
  if ! docker image inspect "${proto_generated_image}" >/dev/null 2>&1; then
    echo "Error: Base image ${proto_generated_image} not found. Build proto intermediate first." >&2
    exit 1
  fi
  
  local build_args=(
    "--target" "artifacts"
    "--build-arg" "BASE_REGISTRY=${IMAGE_BASE}"
    "--build-arg" "BASE_TAG=latest"
    "--build-arg" "PROTO_VERSION=${proto_version}"
  )

  if [[ -n "${BAZEL_REMOTE_CACHE:-}" ]]; then
    build_args+=("--build-arg" "BAZEL_REMOTE_CACHE=${BAZEL_REMOTE_CACHE}")
    if [[ -n "${BAZEL_REMOTE_UPLOAD_LOCAL_RESULTS:-}" ]]; then
      build_args+=("--build-arg" "BAZEL_REMOTE_UPLOAD_LOCAL_RESULTS=${BAZEL_REMOTE_UPLOAD_LOCAL_RESULTS}")
    fi
  fi

  print_stage_header "Building C++ intermediate (${cpp_version})"
  
  local docker_build_args=()
  if [[ "${REGISTRY}" != "local" ]]; then
    docker_build_args+=("--pull")
  fi
  docker_build_args+=("--no-cache")
  
  docker build \
    "${docker_build_args[@]}" \
    --build-arg "TARGETARCH=${TARGETARCH}" \
    "${build_args[@]}" \
    -f "cpp_accelerator/Dockerfile.build" \
    -t "${version_tag}" \
    -t "${latest_tag}" \
    "${REPO_ROOT}"
  
  docker image inspect "${version_tag}" >/dev/null 2>&1

  if [[ "${REGISTRY}" != "local" ]]; then
    echo "Pushing ${version_tag}..."
    docker push "${version_tag}" || true
    echo "Pushing ${latest_tag}..."
    docker push "${latest_tag}" || true
  fi
}

run_golang_built() {
  local proto_version
  local golang_version
  proto_version="$(read_version "proto/VERSION")"
  golang_version="$(read_version "webserver/VERSION")"
  local version_tag="${IMAGE_BASE}/intermediate:golang-built-${golang_version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/intermediate:golang-built-latest-${ARCH}"

  print_stage_header "Building Golang intermediate (${golang_version})"
  build_and_tag \
    "${version_tag}" \
    "${latest_tag}" \
    "webserver/Dockerfile.build" \
    "true" \
    "--target" "artifacts" \
    "--build-arg" "BASE_REGISTRY=${IMAGE_BASE}" \
    "--build-arg" "BASE_TAG=latest" \
    "--build-arg" "PROTO_VERSION=${proto_version}"
}

run_app_image() {
  local proto_version
  local cpp_version
  local golang_version
  proto_version="$(read_version "proto/VERSION")"
  cpp_version="$(read_version "cpp_accelerator/VERSION")"
  golang_version="$(read_version "webserver/VERSION")"
  local app_tag="${golang_version}"
  local version_tag="${IMAGE_BASE}/app:${app_tag}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/app:latest-${ARCH}"

  print_stage_header "Building application image (${app_tag})"
  build_and_tag \
    "${version_tag}" \
    "${latest_tag}" \
    "Dockerfile" \
    "false" \
    "--build-arg" "BASE_REGISTRY=${IMAGE_BASE}" \
    "--build-arg" "BASE_TAG=latest" \
    "--build-arg" "PROTO_VERSION=${proto_version}" \
    "--build-arg" "CPP_VERSION=${cpp_version}" \
    "--build-arg" "GOLANG_VERSION=${golang_version}"
}

run_grpc_server_image() {
  local proto_version
  local cpp_version
  proto_version="$(read_version "proto/VERSION")"
  cpp_version="$(read_version "cpp_accelerator/VERSION")"

  local app_tag="grpc-${cpp_version}-proto${proto_version}"
  local version_tag="${IMAGE_BASE}/grpc-server:${app_tag}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/grpc-server:latest-${ARCH}"

  print_stage_header "Building gRPC server image (${app_tag})"

  local cpp_built_image="${IMAGE_BASE}/intermediate:cpp-built-latest-${ARCH}"
  local proto_generated_image="${IMAGE_BASE}/intermediate:proto-generated-${proto_version}-${ARCH}"
  
  if ! docker image inspect "${proto_generated_image}" >/dev/null 2>&1; then
    echo "Error: Base image ${proto_generated_image} not found. Build proto intermediate first." >&2
    exit 1
  fi
  
  if ! docker image inspect "${cpp_built_image}" >/dev/null 2>&1; then
    echo "Error: Base image ${cpp_built_image} not found. Build cpp intermediate first." >&2
    exit 1
  fi

  local build_args=(
    "--target" "grpc-server"
    "--build-arg" "BASE_REGISTRY=${IMAGE_BASE}"
    "--build-arg" "BASE_TAG=latest"
    "--build-arg" "PROTO_VERSION=${proto_version}"
  )

  if [[ -n "${BAZEL_REMOTE_CACHE:-}" ]]; then
    build_args+=("--build-arg" "BAZEL_REMOTE_CACHE=${BAZEL_REMOTE_CACHE}")
    if [[ -n "${BAZEL_REMOTE_UPLOAD_LOCAL_RESULTS:-}" ]]; then
      build_args+=("--build-arg" "BAZEL_REMOTE_UPLOAD_LOCAL_RESULTS=${BAZEL_REMOTE_UPLOAD_LOCAL_RESULTS}")
    fi
  fi

  build_and_tag \
    "${version_tag}" \
    "${latest_tag}" \
    "cpp_accelerator/Dockerfile.build" \
    "false" \
    "${build_args[@]}" \
    "--build-arg" "TARGETARCH=${TARGETARCH}"
}

for stage in "${REQUESTED_STAGES[@]}"; do
  case "${stage}" in
    proto-tools)
      run_proto_tools
      ;;
    go-builder)
      run_go_builder
      ;;
    bazel-base)
      run_bazel_base
      ;;
    runtime-base)
      run_runtime_base
      ;;
    integration-base)
      run_integration_base
      ;;
    proto)
      run_proto_generated
      ;;
    cpp)
      run_cpp_built
      ;;
    golang)
      run_golang_built
      ;;
    app)
      run_app_image
      ;;
    grpc-server)
      run_grpc_server_image
      ;;
    *)
      echo "Stage '${stage}' is not implemented" >&2
      exit 1
      ;;
  esac
done

echo ""
echo "All requested stages completed."
echo ""


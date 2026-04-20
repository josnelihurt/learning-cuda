#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REGISTRY="${REGISTRY:-local}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt-code/learning-cuda}"
ARCH_DEFAULT="$(uname -m)"
case "${ARCH_DEFAULT}" in
  x86_64) ARCH_DEFAULT="amd64" ;;
  aarch64) ARCH_DEFAULT="arm64" ;;
esac
ARCH="${ARCH_DEFAULT}"
REQUESTED_STAGES=()
SOURCE_REPO_URL="https://github.com/josnelihurt-code/learning-cuda"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build local Docker images sequentially using docker build.

Options:
  --arch <arch>      Target architecture (amd64 or arm64). Defaults to host arch.
  --registry <name>  Registry prefix for tags. Defaults to value of \$REGISTRY or "local".
  --base-prefix <p>  Image namespace following the registry. Defaults to "josnelihurt-code/learning-cuda".
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

ALL_STAGES=(proto-tools go-builder bazel-base runtime-base integration-base proto cpp golang app cpp-accelerator web-frontend)

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
  # Self-hosted pools may sit on different LANs (e.g. prox4 vs prox3); try each bazel-remote
  # HTTP status endpoint until one responds. Override with BAZEL_REMOTE_CACHE_CANDIDATE_HOSTS
  # (space-separated) or set BAZEL_REMOTE_CACHE directly.
  read -r -a _bazel_cache_hosts <<< "${BAZEL_REMOTE_CACHE_CANDIDATE_HOSTS:-192.168.10.80 192.168.30.60}"
  _bazel_cache_found=""
  for _h in "${_bazel_cache_hosts[@]}"; do
    if [[ -z "${_h}" ]]; then
      continue
    fi
    if curl -sf --connect-timeout 2 --max-time 4 "http://${_h}:9090/status" >/dev/null 2>&1; then
      export BAZEL_REMOTE_CACHE="grpc://${_h}:9092"
      export BAZEL_REMOTE_UPLOAD_LOCAL_RESULTS="true"
      echo "Detected bazel-remote cache at ${_h}"
      _bazel_cache_found=1
      break
    fi
  done
  if [[ -z "${_bazel_cache_found}" ]]; then
    echo "Bazel-remote cache not reachable at ${_bazel_cache_hosts[*]}; proceeding without remote cache."
  fi
  unset _h _bazel_cache_hosts _bazel_cache_found
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

  echo "------------------------------------------"
  echo "docker build -f ${dockerfile}"
  echo "  Tag (versioned): ${tag}"
  echo "  Tag (latest):    ${latest_tag}"
  echo "------------------------------------------"

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
    --label "org.opencontainers.image.source=${SOURCE_REPO_URL}" \
    --label "org.opencontainers.image.url=${SOURCE_REPO_URL}" \
    --label "org.opencontainers.image.title=learning-cuda" \
    "${build_args[@]}" \
    -f "${dockerfile}" \
    -t "${tag}" \
    -t "${latest_tag}" \
    "${REPO_ROOT}"

  docker image inspect "${tag}" >/dev/null 2>&1

  if [[ "${should_push}" == "true" && "${REGISTRY}" != "local" ]]; then
    echo "Pushing ${tag}..."
    docker push "${tag}"
    echo "Pushing ${latest_tag}..."
    docker push "${latest_tag}"
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
  version="$(read_version "src/go_api/builder/VERSION")"
  local version_tag="${IMAGE_BASE}/base:go-builder-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/base:go-builder-latest-${ARCH}"

  print_stage_header "Building go builder base (${version})"
  build_and_tag "${version_tag}" "${latest_tag}" "src/go_api/builder/Dockerfile" "true"
}

run_bazel_base() {
  local version
  version="$(read_version "src/cpp_accelerator/docker-build-base/VERSION")"
  local version_tag="${IMAGE_BASE}/base:bazel-base-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/base:bazel-base-latest-${ARCH}"

  print_stage_header "Building bazel base (${version})"
  build_and_tag "${version_tag}" "${latest_tag}" "src/cpp_accelerator/docker-build-base/Dockerfile" "true"
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
  version="$(read_version "test/integration/VERSION")"
  local version_tag="${IMAGE_BASE}/base:integration-tests-base-${version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/base:integration-tests-base-latest-${ARCH}"

  print_stage_header "Building integration base (${version})"
  build_and_tag "${version_tag}" "${latest_tag}" "test/integration/Dockerfile" "true"
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
  cpp_version="$(read_version "src/cpp_accelerator/VERSION")"
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
    -f "src/cpp_accelerator/Dockerfile.build" \
    -t "${version_tag}" \
    -t "${latest_tag}" \
    "${REPO_ROOT}"
  
  docker image inspect "${version_tag}" >/dev/null 2>&1

  if [[ "${REGISTRY}" != "local" ]]; then
    echo "Pushing ${version_tag}..."
    docker push "${version_tag}"
    echo "Pushing ${latest_tag}..."
    docker push "${latest_tag}"
  fi
}

run_golang_built() {
  local proto_version
  local golang_version
  proto_version="$(read_version "proto/VERSION")"
  golang_version="$(read_version "src/go_api/VERSION")"
  local version_tag="${IMAGE_BASE}/intermediate:golang-built-${golang_version}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/intermediate:golang-built-latest-${ARCH}"

  print_stage_header "Building Golang intermediate (${golang_version})"
  build_and_tag \
    "${version_tag}" \
    "${latest_tag}" \
    "src/go_api/Dockerfile.build" \
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
  cpp_version="$(read_version "src/cpp_accelerator/VERSION")"
  golang_version="$(read_version "src/go_api/VERSION")"
  local app_tag="${golang_version}"
  local version_tag="${IMAGE_BASE}/app:${app_tag}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/app:latest-${ARCH}"

  print_stage_header "Building application image (${app_tag})"
  build_and_tag \
    "${version_tag}" \
    "${latest_tag}" \
    "Dockerfile" \
    "false" \
    "--target" "runtime" \
    "--build-arg" "BASE_REGISTRY=${IMAGE_BASE}" \
    "--build-arg" "BASE_TAG=latest" \
    "--build-arg" "PROTO_VERSION=${proto_version}" \
    "--build-arg" "CPP_VERSION=${cpp_version}" \
    "--build-arg" "GOLANG_VERSION=${golang_version}"
}

run_cpp_accelerator_image() {
  local proto_version
  local cpp_version
  proto_version="$(read_version "proto/VERSION")"
  cpp_version="$(read_version "src/cpp_accelerator/VERSION")"

  local app_tag="cpp-accelerator-${cpp_version}-proto${proto_version}"
  local version_tag="${IMAGE_BASE}/cpp-accelerator:${app_tag}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/cpp-accelerator:latest-${ARCH}"

  print_stage_header "Building cpp-accelerator image (${app_tag})"

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
    "--target" "cpp-accelerator"
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
    "src/cpp_accelerator/Dockerfile.build" \
    "false" \
    "${build_args[@]}" \
    "--build-arg" "TARGETARCH=${TARGETARCH}"
}

run_web_frontend_image() {
  local proto_version
  proto_version="$(read_version "proto/VERSION")"

  local fe_version
  fe_version="$(read_version "src/front-end/VERSION")"

  local app_tag="fe-${fe_version}-proto${proto_version}"
  local version_tag="${IMAGE_BASE}/web-frontend:${app_tag}-${ARCH}"
  local latest_tag="${IMAGE_BASE}/web-frontend:latest-${ARCH}"

  print_stage_header "Building web-frontend image (${app_tag})"

  local proto_tools_image="${IMAGE_BASE}/base:proto-tools-latest-${ARCH}"
  if ! docker image inspect "${proto_tools_image}" >/dev/null 2>&1; then
    echo "Error: Base image ${proto_tools_image} not found. Build proto-tools first." >&2
    exit 1
  fi

  build_and_tag \
    "${version_tag}" \
    "${latest_tag}" \
    "src/front-end/Dockerfile" \
    "false" \
    "--build-arg" "BASE_REGISTRY=${IMAGE_BASE}" \
    "--build-arg" "BASE_TAG=latest"
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
    cpp-accelerator)
      run_cpp_accelerator_image
      ;;
    web-frontend)
      run_web_frontend_image
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

if [[ "${REGISTRY}" != "local" ]]; then
  echo "To push all tagged images for ${IMAGE_BASE}, run:"
  echo "  REGISTRY=${REGISTRY} BASE_IMAGE_PREFIX=${BASE_IMAGE_PREFIX} ./scripts/docker/push-tagged-images.sh"
  echo ""
fi


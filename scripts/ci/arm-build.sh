#!/usr/bin/env bash
#
# ARM CI orchestrator for cpp-accelerator pipeline.
#
# Replaces the per-bucket PR jobs and the inline build-and-push shell that used
# to live in .github/workflows/docker-monorepo-build-arm.yml. One job calls this
# script with --mode pr or --mode push; the change-detection booleans decide
# which intermediate stages get rebuilt locally vs pulled from GHCR.
#
# Usage:
#   MODE=pr|push  (or --mode <pr|push>)
#   BUILD_PROTO=0|1         proto/ or buf* changed
#   BUILD_BAZEL_BASE=0|1    src/cpp_accelerator/docker-build-base/** changed
#   BUILD_CPP_DEPS=0|1      bazel/, third_party/, MODULE.bazel*, .bazelrc, docker-cpp-dependencies/** changed
#                           (BUILD_BAZEL_BASE=1 implies BUILD_CPP_DEPS=1; the workflow already applies this)
#   BUILD_CUDA_RUNTIME=0|1  src/cpp_accelerator/docker-cuda-runtime/** changed
#
# The script always builds cpp-builder (it depends on workspace HEAD).
# In --mode push it also builds cpp-accelerator and pushes only what was rebuilt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${MODE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    -h|--help)
      sed -n '3,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "${MODE}" != "pr" && "${MODE}" != "push" ]]; then
  echo "MODE must be 'pr' or 'push' (got '${MODE}')" >&2
  exit 1
fi

BUILD_PROTO="${BUILD_PROTO:-0}"
BUILD_BAZEL_BASE="${BUILD_BAZEL_BASE:-0}"
BUILD_CPP_DEPS="${BUILD_CPP_DEPS:-0}"
BUILD_CUDA_RUNTIME="${BUILD_CUDA_RUNTIME:-0}"

REGISTRY="${REGISTRY:-ghcr.io}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt-code/learning-cuda}"
ARCH="${ARCH:-arm64}"

# Suppress build-local.sh's per-stage auto-push. We want this script to own
# every push decision so cpp-builder (and any merely-rebuilt PR intermediates)
# don't leak to GHCR.
export BUILD_LOCAL_PUSH=0

# Suppress build-local.sh's `docker build --pull`. Forced pulls fail when an
# intermediate has been built locally but not yet pushed (e.g. a PR that bumps
# proto/VERSION — the new versioned tag doesn't exist on GHCR yet). The
# orchestrator already explicitly pulled what's needed.
export BUILD_LOCAL_PULL=0

BL=("${ROOT}/scripts/docker/build-local.sh"
    --registry "${REGISTRY}"
    --base-prefix "${BASE_IMAGE_PREFIX}"
    --arch "${ARCH}")

chmod +x "${ROOT}/scripts/docker/build-local.sh" \
         "${ROOT}/scripts/docker/pull-ghcr-cpp-intermediates.sh"

echo "=== arm-build.sh: MODE=${MODE} ARCH=${ARCH}"
echo "    BUILD_PROTO=${BUILD_PROTO} BUILD_BAZEL_BASE=${BUILD_BAZEL_BASE} BUILD_CPP_DEPS=${BUILD_CPP_DEPS} BUILD_CUDA_RUNTIME=${BUILD_CUDA_RUNTIME}"

# 1. Build the intermediates whose inputs changed. The order matches the
#    Dockerfile.build dependency chain.
if [[ "${BUILD_PROTO}" == "1" ]]; then
  "${BL[@]}" --stage proto-tools --stage proto
fi

if [[ "${BUILD_CPP_DEPS}" == "1" ]]; then
  # cpp-dependencies depends on bazel-base, so always rebuild bazel-base first.
  "${BL[@]}" --stage bazel-base --stage cpp-dependencies
elif [[ "${BUILD_BAZEL_BASE}" == "1" ]]; then
  # Defensive: workflow should have set BUILD_CPP_DEPS=1 in this case, but if
  # only bazel-base is flagged we still need its downstream deps layer to match.
  "${BL[@]}" --stage bazel-base --stage cpp-dependencies
fi

if [[ "${BUILD_CUDA_RUNTIME}" == "1" ]]; then
  "${BL[@]}" --stage cuda-runtime
fi

# 2. Pull whatever we did NOT rebuild from GHCR. The pull script no-ops if the
#    image is already present locally.
PULL_PROTO_LATEST=0
PULL_CPP_DEPENDENCIES=0
PULL_CUDA_RUNTIME=0
[[ "${BUILD_PROTO}" == "1" ]] || PULL_PROTO_LATEST=1
[[ "${BUILD_CPP_DEPS}" == "1" || "${BUILD_BAZEL_BASE}" == "1" ]] || PULL_CPP_DEPENDENCIES=1
[[ "${BUILD_CUDA_RUNTIME}" == "1" ]] || PULL_CUDA_RUNTIME=1

PULL_PROTO_LATEST="${PULL_PROTO_LATEST}" \
  PULL_CPP_DEPENDENCIES="${PULL_CPP_DEPENDENCIES}" \
  PULL_CUDA_RUNTIME="${PULL_CUDA_RUNTIME}" \
  "${ROOT}/scripts/docker/pull-ghcr-cpp-intermediates.sh"

# 3. Always build cpp-builder. It compiles the C++ from the workspace HEAD,
#    which is the only thing that's guaranteed to be different on every run.
"${BL[@]}" --stage cpp-builder

if [[ "${MODE}" == "pr" ]]; then
  echo "=== arm-build.sh: PR mode complete (no runtime, no push)."
  exit 0
fi

# 4. Push mode: build the runtime image (just COPYs from cpp-builder) and push
#    only the images we actually want to publish. We avoid push-tagged-images.sh
#    here because its bulk-push would also publish cpp-builder and re-publish
#    pulled intermediates — both wasteful.
"${BL[@]}" --stage cpp-accelerator

read_version() {
  local path="$1"
  if [[ ! -f "${ROOT}/${path}" ]]; then
    echo "Version file '${path}' not found" >&2; exit 1
  fi
  tr -d '[:space:]' < "${ROOT}/${path}"
}

IMAGE_BASE="${REGISTRY}/${BASE_IMAGE_PREFIX}"

push_ref() {
  local ref="$1"
  if ! docker image inspect "${ref}" >/dev/null 2>&1; then
    echo "ERROR: expected image ${ref} not present locally" >&2
    exit 1
  fi
  echo "Pushing ${ref}..."
  docker push "${ref}"
}

# Always: cpp-accelerator runtime image (versioned + latest alias).
proto_version="$(read_version proto/VERSION)"
cpp_version="$(read_version src/cpp_accelerator/VERSION)"
push_ref "${IMAGE_BASE}/cpp-accelerator:cpp-accelerator-${cpp_version}-proto${proto_version}-${ARCH}"
push_ref "${IMAGE_BASE}/cpp-accelerator:latest-${ARCH}"

# Conditionally: intermediates we just rebuilt. Pulled images are NOT re-pushed.
if [[ "${BUILD_PROTO}" == "1" ]]; then
  push_ref "${IMAGE_BASE}/intermediate:proto-generated-${proto_version}-${ARCH}"
  push_ref "${IMAGE_BASE}/intermediate:proto-generated-latest-${ARCH}"
fi

if [[ "${BUILD_BAZEL_BASE}" == "1" ]]; then
  bazel_base_version="$(read_version src/cpp_accelerator/docker-build-base/VERSION)"
  push_ref "${IMAGE_BASE}/base:bazel-base-${bazel_base_version}-${ARCH}"
  push_ref "${IMAGE_BASE}/base:bazel-base-latest-${ARCH}"
fi

if [[ "${BUILD_CPP_DEPS}" == "1" || "${BUILD_BAZEL_BASE}" == "1" ]]; then
  deps_version="$(read_version src/cpp_accelerator/docker-cpp-dependencies/VERSION)"
  push_ref "${IMAGE_BASE}/intermediate:cpp-dependencies-${deps_version}-${ARCH}"
  push_ref "${IMAGE_BASE}/intermediate:cpp-dependencies-latest-${ARCH}"
fi

if [[ "${BUILD_CUDA_RUNTIME}" == "1" ]]; then
  cuda_runtime_version="$(read_version src/cpp_accelerator/docker-cuda-runtime/VERSION)"
  push_ref "${IMAGE_BASE}/base:cuda-runtime-${cuda_runtime_version}-${ARCH}"
  push_ref "${IMAGE_BASE}/base:cuda-runtime-latest-${ARCH}"
fi

echo "=== arm-build.sh: push mode complete."

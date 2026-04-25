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
#   BUILD_PROTO=0|1       proto/ or buf* changed
#   BUILD_BAZEL_BASE=0|1  src/cpp_accelerator/docker-build-base/** changed
#   BUILD_CPP_DEPS=0|1    bazel/, third_party/, MODULE.bazel*, .bazelrc, docker-cpp-dependencies/** changed
#                         (BUILD_BAZEL_BASE=1 implies BUILD_CPP_DEPS=1; the workflow already applies this)
#
# The script always builds cpp-builder (it depends on workspace HEAD).
# In --mode push it also builds cpp-accelerator and runs push-tagged-images.sh.

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

REGISTRY="${REGISTRY:-ghcr.io}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt-code/learning-cuda}"
ARCH="${ARCH:-arm64}"

BL=("${ROOT}/scripts/docker/build-local.sh"
    --registry "${REGISTRY}"
    --base-prefix "${BASE_IMAGE_PREFIX}"
    --arch "${ARCH}")

chmod +x "${ROOT}/scripts/docker/build-local.sh" \
         "${ROOT}/scripts/docker/pull-ghcr-cpp-intermediates.sh" \
         "${ROOT}/scripts/docker/push-tagged-images.sh"

echo "=== arm-build.sh: MODE=${MODE} ARCH=${ARCH}"
echo "    BUILD_PROTO=${BUILD_PROTO} BUILD_BAZEL_BASE=${BUILD_BAZEL_BASE} BUILD_CPP_DEPS=${BUILD_CPP_DEPS}"

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

# 2. Pull whatever we did NOT rebuild from GHCR. The pull script no-ops if the
#    image is already present locally.
PULL_PROTO_LATEST=0
PULL_CPP_DEPENDENCIES=0
[[ "${BUILD_PROTO}" == "1" ]] || PULL_PROTO_LATEST=1
[[ "${BUILD_CPP_DEPS}" == "1" || "${BUILD_BAZEL_BASE}" == "1" ]] || PULL_CPP_DEPENDENCIES=1

PULL_PROTO_LATEST="${PULL_PROTO_LATEST}" PULL_CPP_DEPENDENCIES="${PULL_CPP_DEPENDENCIES}" \
  "${ROOT}/scripts/docker/pull-ghcr-cpp-intermediates.sh"

# 3. Always build cpp-builder. It compiles the C++ from the workspace HEAD,
#    which is the only thing that's guaranteed to be different on every run.
"${BL[@]}" --stage cpp-builder

if [[ "${MODE}" == "pr" ]]; then
  echo "=== arm-build.sh: PR mode complete (no runtime, no push)."
  exit 0
fi

# 4. Push mode: build the runtime image (just COPYs from cpp-builder) and push
#    everything tagged under our prefix. push-tagged-images.sh bulk-pushes any
#    locally tagged image, so newly built intermediates land on GHCR with both
#    versioned and latest-${ARCH} tags. Pulled images get re-pushed too, which
#    is a content-identical no-op.
"${BL[@]}" --stage cpp-accelerator

LATEST_ALIASES="cpp-accelerator" \
  "${ROOT}/scripts/docker/push-tagged-images.sh"

echo "=== arm-build.sh: push mode complete."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

ORDER=(proto-tools bazel-base cpp-dependencies proto cpp-builder)
declare -A WANT=()

if [[ "${PR_ARM_FULL:-false}" == "true" ]]; then
  for s in "${ORDER[@]}"; do
    WANT["$s"]=1
  done
else
  if [[ "${ARM_PROTO:-false}" == "true" ]]; then
    WANT[proto-tools]=1
    WANT[proto]=1
    WANT[cpp-builder]=1
  fi
  if [[ "${ARM_BAZEL_BASE:-false}" == "true" ]]; then
    WANT[bazel-base]=1
    WANT[cpp-dependencies]=1
    WANT[cpp-builder]=1
  fi
  if [[ "${ARM_CPP_DEPS:-false}" == "true" ]]; then
    WANT[bazel-base]=1
    WANT[cpp-dependencies]=1
    WANT[cpp-builder]=1
  fi
  if [[ "${ARM_CPP_APP:-false}" == "true" ]]; then
    WANT[cpp-builder]=1
  fi
  if [[ "${ARM_WORKFLOW:-false}" == "true" ]]; then
    for s in "${ORDER[@]}"; do
      WANT["$s"]=1
    done
  fi
fi

args=()
for s in "${ORDER[@]}"; do
  [[ -n "${WANT[$s]:-}" ]] && args+=(--stage "$s")
done

if [[ ${#args[@]} -eq 0 ]]; then
  echo "pr-arm-resolve-stages: no stages selected" >&2
  exit 1
fi

chmod +x "${ROOT}/scripts/docker/build-local.sh"
exec "${ROOT}/scripts/docker/build-local.sh" "${args[@]}"

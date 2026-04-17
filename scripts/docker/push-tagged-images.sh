#!/usr/bin/env bash
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt-code/learning-cuda}"
IMAGE_PREFIX="${REGISTRY}/${BASE_IMAGE_PREFIX}"
# Must match build-local.sh (amd64 on x86 CI, arm64 on ARM CI). Used for explicit latest-* pushes.
ARCH="${ARCH:-amd64}"

require_command() {
  local cmd="$1"
  local hint="${2:-}"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command '${cmd}' not found in PATH." >&2
    if [[ -n "${hint}" ]]; then
      echo "Install hint: ${hint}" >&2
    fi
    exit 1
  fi
}

require_command docker
require_command rg "Debian/Ubuntu: sudo apt-get install -y ripgrep"
require_command sort
require_command head

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not available" >&2
  exit 1
fi

# Listed before any pipeline so a docker failure isn't masked by the pipe.
all_local_images="$(docker images --format '{{.Repository}}:{{.Tag}}')"

images_to_push="$(printf '%s\n' "${all_local_images}" \
  | rg "^${IMAGE_PREFIX}/" \
  | rg -v ":(<none>|none)$" \
  || true)"

if [[ -z "${images_to_push}" ]]; then
  echo "No local images found for prefix ${IMAGE_PREFIX}/" >&2
  echo "Nothing to push. Did the build stage actually produce images for this prefix?" >&2
  echo "--- Local image inventory (head) ---" >&2
  printf '%s\n' "${all_local_images}" | head -n 30 >&2
  exit 1
fi

pushed=0
while IFS= read -r img; do
  [[ -z "${img}" ]] && continue
  echo "Pushing ${img}..."
  docker push "${img}"
  pushed=$((pushed + 1))
done <<< "${images_to_push}"

# The bulk loop above can still miss publishing latest-${ARCH} to GHCR (e.g. only the versioned
# tag was tagged locally, or tooling listed one ref per image). Always publish canonical aliases.
echo ""
echo "Publishing latest-${ARCH} aliases for app, grpc-server, web-frontend..."

failed_aliases=()

publish_latest_alias() {
  local name="$1"
  local latest_ref="${IMAGE_PREFIX}/${name}:latest-${ARCH}"

  if docker image inspect "${latest_ref}" >/dev/null 2>&1; then
    echo "Pushing ${latest_ref}..."
    docker push "${latest_ref}"
    return 0
  fi

  if [[ "${name}" == "web-frontend" ]]; then
    local versioned
    versioned="$(printf '%s\n' "${all_local_images}" \
      | rg "^${IMAGE_PREFIX}/web-frontend:fe-" \
      | sort -Vr \
      | head -1 \
      || true)"
    if [[ -n "${versioned}" ]]; then
      echo "Repairing missing ${latest_ref} from ${versioned}"
      docker tag "${versioned}" "${latest_ref}"
      docker push "${latest_ref}"
      return 0
    fi
  fi

  echo "ERROR: could not publish ${latest_ref} (no local image or fe-* tag to repair from)" >&2
  failed_aliases+=("${latest_ref}")
  return 1
}

# Disable -e for the alias loop so we collect every failure before reporting.
set +e
publish_latest_alias "app"
publish_latest_alias "grpc-server"
publish_latest_alias "web-frontend"
set -e

if [[ "${#failed_aliases[@]}" -gt 0 ]]; then
  echo "" >&2
  echo "Failed to publish ${#failed_aliases[@]} latest-${ARCH} alias(es):" >&2
  printf '  - %s\n' "${failed_aliases[@]}" >&2
  echo "Local images inspected (head):" >&2
  printf '%s\n' "${all_local_images}" | head -n 30 >&2
  exit 1
fi

echo "Done. Pushed ${pushed} image(s) in the bulk pass; latest-${ARCH} aliases handled explicitly."

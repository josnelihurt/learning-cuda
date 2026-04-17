#!/usr/bin/env bash
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt-code/learning-cuda}"
IMAGE_PREFIX="${REGISTRY}/${BASE_IMAGE_PREFIX}"
# Must match build-local.sh (amd64 on x86 CI, arm64 on ARM CI). Used for explicit latest-* pushes.
ARCH="${ARCH:-amd64}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not available" >&2
  exit 1
fi

images_to_push="$(
  docker images --format "{{.Repository}}:{{.Tag}}" \
    | rg "^${IMAGE_PREFIX}/" \
    | rg -v ":(<none>|none)$" \
    || true
)"

if [[ -z "${images_to_push}" ]]; then
  echo "No local images found for prefix ${IMAGE_PREFIX}/"
  exit 0
fi

pushed=0
while IFS= read -r img; do
  [[ -z "${img}" ]] && continue
  echo "Pushing ${img}..."
  docker push "${img}"
  ((pushed++))
done <<< "${images_to_push}"

# The bulk loop above can still miss publishing latest-${ARCH} to GHCR (e.g. only the versioned
# tag was tagged locally, or tooling listed one ref per image). Always publish canonical aliases.
echo ""
echo "Publishing latest-${ARCH} aliases for app, grpc-server, web-frontend..."
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
    versioned="$(docker images --format "{{.Repository}}:{{.Tag}}" \
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

  echo "Warning: could not publish ${latest_ref} (no local image or fe-* tag to repair from)" >&2
}

publish_latest_alias "app"
publish_latest_alias "grpc-server"
publish_latest_alias "web-frontend"

echo "Done. Pushed ${pushed} image(s) in the bulk pass; latest-${ARCH} aliases handled explicitly."

#!/usr/bin/env bash
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io}"
BASE_IMAGE_PREFIX="${BASE_IMAGE_PREFIX:-josnelihurt-code/learning-cuda}"
IMAGE_PREFIX="${REGISTRY}/${BASE_IMAGE_PREFIX}"

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

echo "Done. Pushed ${pushed} image(s)."

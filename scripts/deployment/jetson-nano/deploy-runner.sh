#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${SCRIPT_DIR}/test.sh"

for cmd in gh ssh; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
done

REPO_URL="$(git -C "${PROJECT_ROOT}" config --get remote.origin.url)"
REPO_SLUG="${REPO_URL#*:}"
REPO_SLUG="${REPO_SLUG%.git}"
if [[ "${REPO_SLUG}" == "${REPO_URL}" ]]; then
  REPO_SLUG=$(echo "${REPO_URL}" | sed -E 's#https?://github.com/([^/]+/[^/]+)(\.git)?#\1#')
fi

if [[ -z "${REPO_SLUG}" ]]; then
  echo "Unable to determine repository slug from ${REPO_URL}" >&2
  exit 1
fi

RUNNER_NAME="learning-cuda-jetson-nano-1"
RUNNER_CONTAINER="learning-cuda-runner"
RUNNER_IMAGE="${RUNNER_IMAGE:-ghcr.io/actions/actions-runner:latest}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,Linux,ARM64,jetson,jetson-nano}"
JETSON_TARGET="${JETSON_USER}@${JETSON_HOST}"

ensure_runner_removed() {
  local runner_id
  runner_id="$(gh api "repos/${REPO_SLUG}/actions/runners" --paginate --jq ".runners[] | select(.name==\"${RUNNER_NAME}\") | .id" || true)"
  if [[ -n "${runner_id}" ]]; then
    echo "Removing existing runner registration (${runner_id})..."
    gh api "repos/${REPO_SLUG}/actions/runners/${runner_id}" --method DELETE >/dev/null
  fi
}

stop_runner() {
  echo "Stopping GitHub runner container on Jetson Nano..."
  ssh "${JETSON_TARGET}" "docker stop ${RUNNER_CONTAINER} >/dev/null 2>&1 || true"
  ssh "${JETSON_TARGET}" "docker rm ${RUNNER_CONTAINER} >/dev/null 2>&1 || true"
  ensure_runner_removed
  echo "Runner '${RUNNER_NAME}' stopped and deregistered."
}

start_runner() {
  ensure_runner_removed
  echo "Requesting GitHub Actions registration token..."
  local registration_token
  registration_token="$(gh api --method POST "repos/${REPO_SLUG}/actions/runners/registration-token" --jq .token)"

  echo "Pulling runner image on Jetson Nano (${RUNNER_IMAGE})..."
  ssh "${JETSON_TARGET}" "docker pull ${RUNNER_IMAGE}"

  echo "Launching runner container '${RUNNER_CONTAINER}'..."
  ssh "${JETSON_TARGET}" bash <<EOF
set -euo pipefail
docker stop ${RUNNER_CONTAINER} >/dev/null 2>&1 || true
docker rm ${RUNNER_CONTAINER} >/dev/null 2>&1 || true
docker run -d --restart always \\
  --name ${RUNNER_CONTAINER} \\
  -e ACTIONS_RUNNER_NAME="${RUNNER_NAME}" \\
  -e ACTIONS_RUNNER_REPO_URL="https://github.com/${REPO_SLUG}" \\
  -e ACTIONS_RUNNER_TOKEN="${registration_token}" \\
  -e ACTIONS_RUNNER_LABELS="${RUNNER_LABELS}" \\
  -e ACTIONS_RUNNER_WORKDIR="/tmp/actions-runner" \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  ${RUNNER_IMAGE}
EOF

  echo "Waiting for runner to come online..."
  for attempt in {1..12}; do
    status="$(gh api "repos/${REPO_SLUG}/actions/runners" --paginate --jq ".runners[] | select(.name==\"${RUNNER_NAME}\") | .status" || true)"
    if [[ "${status}" == "online" ]]; then
      echo "Runner '${RUNNER_NAME}' is online."
      return 0
    fi
    echo "Runner not yet online (attempt ${attempt}/12). Retrying in 10s..."
    sleep 10
  done

  echo "Runner '${RUNNER_NAME}' did not report online status in time." >&2
  exit 1
}

ACTION="${1:-start}"
case "${ACTION}" in
  --stop|stop)
    stop_runner
    ;;
  start|--start|"")
    start_runner
    ;;
  *)
    echo "Usage: $0 [--stop]" >&2
    exit 1
    ;;
esac


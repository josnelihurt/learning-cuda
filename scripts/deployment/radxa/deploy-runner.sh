#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${SCRIPT_DIR}/test.sh"

for cmd in gh ssh ansible-playbook; do
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

RUNNER_NAME="learning-cuda-radxa-1"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,Linux,ARM64,radxa}"
RUNNER_VERSION="${RUNNER_VERSION:-$(gh api repos/actions/runner/releases/latest --jq .tag_name)}"
RUNNER_VERSION_NUMBER="${RUNNER_VERSION#v}"
RUNNER_DOWNLOAD_URL="${RUNNER_DOWNLOAD_URL:-https://github.com/actions/runner/releases/download/${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION_NUMBER}.tar.gz}"

run_ansible() {
  local state="$1"
  ansible-playbook \
    -i "${SCRIPT_DIR}/ansible/inventory.yml" \
    "${SCRIPT_DIR}/ansible/runner.yml" \
    -e "runner_state=${state}" \
    -e "repo_slug=${REPO_SLUG}" \
    -e "runner_name=${RUNNER_NAME}" \
    -e "runner_labels=${RUNNER_LABELS}" \
    -e "runner_download_url=${RUNNER_DOWNLOAD_URL}"
}

wait_for_runner() {
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
    echo "Stopping GitHub runner via Ansible..."
    run_ansible "absent"
    echo "Runner '${RUNNER_NAME}' stopped and deregistered."
    ;;
  start|--start|"")
    echo "Starting GitHub runner via Ansible..."
    run_ansible "present"
    wait_for_runner
    ;;
  *)
    echo "Usage: $0 [--stop]" >&2
    exit 1
    ;;
esac


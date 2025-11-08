#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

for cmd in terraform gh jq; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    echo "Install the tool and rerun this script." >&2
    exit 1
  fi
done

SECRETS_FILE="${PROJECT_ROOT}/.secrets/proxmox-api.key"
if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  exit 1
fi

terraform init -input=false

OUTPUT_JSON_FILE="$(mktemp)"
trap 'rm -f "${OUTPUT_JSON_FILE}"' EXIT

if terraform output -json >"${OUTPUT_JSON_FILE}"; then
  RUNNER_REPO=$(jq -r '.runner_repo.value // empty' "${OUTPUT_JSON_FILE}")
  mapfile -t RUNNER_NAMES < <(jq -r '.runner_names.value[]?' "${OUTPUT_JSON_FILE}")

  if [[ -n "${RUNNER_REPO}" && ${#RUNNER_NAMES[@]} -gt 0 ]]; then
    for RUNNER_NAME in "${RUNNER_NAMES[@]}"; do
    echo "Searching for runner '${RUNNER_NAME}' in ${RUNNER_REPO}..."
    RUNNER_ID=$(gh api "repos/${RUNNER_REPO}/actions/runners" --paginate --jq ".runners[] | select(.name==\"${RUNNER_NAME}\") | .id" || true)

    if [[ -n "${RUNNER_ID}" ]]; then
      echo "Removing GitHub runner '${RUNNER_NAME}' (ID: ${RUNNER_ID})"
      gh api "repos/${RUNNER_REPO}/actions/runners/${RUNNER_ID}" --method DELETE >/dev/null
    else
      echo "Runner '${RUNNER_NAME}' not found in GitHub; skipping removal."
    fi
    done
  fi
else
  echo "Unable to read Terraform outputs; skipping GitHub runner removal." >&2
fi

terraform destroy -input=false -auto-approve "$@"

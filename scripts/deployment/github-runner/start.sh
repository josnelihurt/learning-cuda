#!/usr/bin/env bash

set -euo pipefail

verify_runner_registration() {
  local repo="$1"
  local runner="$2"
  local attempts="${RUNNER_VERIFY_ATTEMPTS:-10}"
  local delay="${RUNNER_VERIFY_DELAY:-15}"

  echo "Validating runner registration for ${runner} in ${repo}..."
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    local status
    status=$(gh api "repos/${repo}/actions/runners" --paginate --jq ".runners[] | select(.name==\"${runner}\") | .status" || true)

    if [[ -n "${status}" ]]; then
      echo "Runner '${runner}' reported status: ${status}"
      if [[ "${status}" == "online" ]]; then
        echo "Runner '${runner}' is online."
        return 0
      fi
    else
      echo "Runner '${runner}' not yet registered (attempt ${attempt}/${attempts})."
    fi

    if [[ ${attempt} -lt ${attempts} ]]; then
      sleep "${delay}"
    fi
  done

  echo "Runner '${runner}' did not register within ${attempts} attempts." >&2
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

for cmd in terraform gh ansible-playbook jq python3; do
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
terraform apply -input=false -auto-approve "$@"

OUTPUT_JSON_FILE="$(mktemp)"
ANSIBLE_VARS_FILE="$(mktemp)"
trap 'rm -f "${OUTPUT_JSON_FILE}" "${ANSIBLE_VARS_FILE}"' EXIT

terraform output -json >"${OUTPUT_JSON_FILE}"

RUNNER_REPO=$(jq -r '.runner_repo.value' "${OUTPUT_JSON_FILE}")
RUNNER_NAME=$(jq -r '.runner_name.value' "${OUTPUT_JSON_FILE}")

export PROXMOX_SSH_USER="${PROXMOX_SSH_USER:-root}"
if [[ -z "${PROXMOX_HOST_TASKS_ENABLED:-}" ]]; then
  if [[ -n "${PROXMOX_SSH_KEY_PATH:-}" ]]; then
    PROXMOX_HOST_TASKS_ENABLED="true"
  else
    PROXMOX_HOST_TASKS_ENABLED="false"
  fi
fi
export PROXMOX_HOST_TASKS_ENABLED
export PROXMOX_SSH_KEY_PATH="${PROXMOX_SSH_KEY_PATH:-}"

if [[ "${PROXMOX_HOST_TASKS_ENABLED}" != "true" ]]; then
  echo "Skipping Proxmox host tasks (set PROXMOX_SSH_KEY_PATH to enable)."
fi

python3 - "${OUTPUT_JSON_FILE}" "${ANSIBLE_VARS_FILE}" <<'PY'
import json
import os
import sys

_, output_path, vars_path = sys.argv

with open(output_path, "r", encoding="utf-8") as fh:
    outputs = json.load(fh)

def get_value(name):
    return outputs[name]["value"]

connection = get_value("runner_connection")
raw_labels = get_value("runner_labels")
if isinstance(raw_labels, str):
    runner_labels = [label.strip() for label in raw_labels.split(",") if label.strip()]
else:
    runner_labels = list(raw_labels)

vars_payload = {
    "proxmox_host": get_value("proxmox_host"),
    "proxmox_user": os.environ.get("PROXMOX_SSH_USER", "root"),
    "proxmox_private_key_path": os.environ.get("PROXMOX_SSH_KEY_PATH") or None,
    "proxmox_host_tasks_enabled": os.environ.get("PROXMOX_HOST_TASKS_ENABLED", "false").lower() in ("1", "true", "yes"),
    "runner_host": connection.get("host"),
    "runner_user": connection.get("user", "root"),
    "runner_private_key_path": connection.get("private_key_path"),
    "runner_repo": get_value("runner_repo"),
    "runner_name": get_value("runner_name"),
    "runner_template": get_value("runner_template"),
    "runner_image_url": get_value("runner_image_url"),
    "runner_labels": runner_labels,
    "runner_labels_csv": ",".join(runner_labels),
    "runner_workdir": "/opt/actions-runner",
}

with open(vars_path, "w", encoding="utf-8") as fh:
    json.dump(vars_payload, fh)
PY

ANSIBLE_DIR="${PROJECT_ROOT}/scripts/deployment/prox4/ansible"
pushd "${ANSIBLE_DIR}" >/dev/null
ansible-playbook site.yml -e "@${ANSIBLE_VARS_FILE}"
popd >/dev/null

verify_runner_registration "${RUNNER_REPO}" "${RUNNER_NAME}"


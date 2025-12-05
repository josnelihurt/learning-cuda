#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Load env
if [ ! -f "${PROJECT_ROOT}/.secrets/production.env" ]; then
    echo "Error: .secrets/production.env not found"
    exit 1
fi
. "${PROJECT_ROOT}/.secrets/production.env"

# Validate vars
for var in CLOUD_VM_HOST CLOUD_VM_USER; do
    eval "value=\$$var"
    if [ -z "$value" ]; then
        echo "Error: $var not set"
        exit 1
    fi
done

# Export for Ansible
export CLOUD_VM_HOST CLOUD_VM_USER

# Check Ansible
ANSIBLE_PATH=$(command -v ansible-playbook 2>/dev/null || echo "")
if [ -z "$ANSIBLE_PATH" ]; then
    echo "Error: Ansible not installed"
    echo "PATH: $PATH"
    exit 1
fi

# Test SSH
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${CLOUD_VM_USER}@${CLOUD_VM_HOST}" "echo OK" > /dev/null 2>&1; then
    echo "Error: SSH connection failed"
    exit 1
fi

echo "All validations passed"


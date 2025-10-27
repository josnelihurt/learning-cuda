#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Load env
if [ ! -f "${PROJECT_ROOT}/.secrets/production.env" ]; then
    echo "Error: .secrets/production.env not found"
    exit 1
fi
source "${PROJECT_ROOT}/.secrets/production.env"

# Validate vars
for var in JETSON_HOST JETSON_USER JETSON_SUDO_PASSWORD JETSON_APP_DIRECTORY; do
    if [ -z "${!var}" ]; then
        echo "Error: $var not set"
        exit 1
    fi
done

# Export for Ansible
export JETSON_HOST JETSON_USER JETSON_SUDO_PASSWORD JETSON_APP_DIRECTORY
export JETSON_DEPLOYMENT_GROUP="${JETSON_DEPLOYMENT_GROUP:-$JETSON_USER}"

# Check Ansible
if ! command -v ansible-playbook &> /dev/null; then
    echo "Error: Ansible not installed"
    exit 1
fi

# Test SSH
if ! ssh -o ConnectTimeout=5 "${JETSON_USER}@${JETSON_HOST}" "echo OK" &> /dev/null; then
    echo "Error: SSH connection failed"
    exit 1
fi

echo "All validations passed"

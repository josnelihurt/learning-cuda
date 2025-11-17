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
# Support both RADXA_* and JETSON_* vars for compatibility
RADXA_HOST="${RADXA_HOST:-${JETSON_HOST:-192.168.10.194}}"
RADXA_USER="${RADXA_USER:-${JETSON_USER}}"
RADXA_SUDO_PASSWORD="${RADXA_SUDO_PASSWORD:-${JETSON_SUDO_PASSWORD}}"
RADXA_APP_DIRECTORY="${RADXA_APP_DIRECTORY:-${JETSON_APP_DIRECTORY}}"

for var in RADXA_USER RADXA_SUDO_PASSWORD RADXA_APP_DIRECTORY; do
    if [ -z "${!var}" ]; then
        echo "Error: $var not set (can also use JETSON_* vars for compatibility)"
        exit 1
    fi
done

# Export for Ansible
export RADXA_HOST RADXA_USER RADXA_SUDO_PASSWORD RADXA_APP_DIRECTORY
export RADXA_DEPLOYMENT_GROUP="${RADXA_DEPLOYMENT_GROUP:-${JETSON_DEPLOYMENT_GROUP:-$RADXA_USER}}"

# Check Ansible
if ! command -v ansible-playbook &> /dev/null; then
    echo "Error: Ansible not installed"
    exit 1
fi

# Test SSH
if ! ssh -o ConnectTimeout=5 "${RADXA_USER}@${RADXA_HOST}" "echo OK" &> /dev/null; then
    echo "Error: SSH connection failed to ${RADXA_USER}@${RADXA_HOST}"
    exit 1
fi

echo "All validations passed"

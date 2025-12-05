#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Load env variables
# Check if variables are already set (CI scenario)
if [ -z "${CLOUD_VM_HOST:-}" ] || [ -z "${CLOUD_VM_USER:-}" ]; then
    # Variables not set, try to load from file (local scenario)
    if [ ! -f "${PROJECT_ROOT}/.secrets/production.env" ]; then
        echo "Error: .secrets/production.env not found and CLOUD_VM_HOST/CLOUD_VM_USER not set"
        exit 1
    fi
    . "${PROJECT_ROOT}/.secrets/production.env"
fi

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

# Test SSH (optional - Ansible will handle connection)
# Skip SSH test in CI/Docker environments where it may not work reliably
# The test is informational only and won't block deployment
if [ -z "${CI:-}" ] && [ -z "${DOCKER_CONTAINER:-}" ] && [ -f ~/.ssh/id_rsa ]; then
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa "${CLOUD_VM_USER}@${CLOUD_VM_HOST}" "echo OK" > /dev/null 2>&1; then
        echo "SSH connection test passed"
    else
        echo "Warning: SSH connection test failed, but continuing (Ansible will handle connection)"
    fi
else
    echo "Skipping SSH test (CI/Docker environment or no SSH key found)"
fi

echo "All validations passed"


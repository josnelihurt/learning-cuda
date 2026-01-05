#!/bin/sh
set -e
APP_IMAGE="ghcr.io/josnelihurt/learning-cuda/app:rc-amd64"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"

# Load env variables
if [ -z "${CLOUD_VM_HOST:-}" ] || [ -z "${CLOUD_VM_USER:-}" ]; then
    if [ ! -f "${PROJECT_ROOT}/.secrets/production.env" ]; then
        echo "Error: .secrets/production.env not found and CLOUD_VM_HOST/CLOUD_VM_USER not set"
        exit 1
    fi
    . "${PROJECT_ROOT}/.secrets/production.env"
fi

export CLOUD_VM_HOST CLOUD_VM_USER CLOUD_VM_SUDO_PASSWORD

# Debug: Show environment variables (without sensitive data)
echo "Deployment configuration:"
echo "  CLOUD_VM_HOST: ${CLOUD_VM_HOST}"
echo "  CLOUD_VM_USER: ${CLOUD_VM_USER}"
echo "  APP_IMAGE: ${APP_IMAGE:-not set}"

if [ "${1:-}" = "--clean" ]; then
    echo "Clean option not implemented yet"
fi

cd "${SCRIPT_DIR}"
export PROJECT_ROOT
echo "Synchronizing files..."
echo "Running: ansible-playbook -i ansible/inventory.yml ansible/sync.yml"
if ! ansible-playbook -i ansible/inventory.yml ansible/sync.yml; then
    echo "ERROR: Ansible playbook failed during file synchronization"
    exit 1
fi

cd "${SCRIPT_DIR}"
echo "Starting services..."
echo "Running: ansible-playbook -i ansible/inventory.yml ansible/start.yml -v"
if ! ansible-playbook -i ansible/inventory.yml ansible/start.yml -v; then
    echo "ERROR: Ansible playbook failed during service startup"
    exit 1
fi


echo "Deployment complete"
echo "Access: https://app-cuda-demo.josnelihurt.me"


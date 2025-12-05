#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "${SCRIPT_DIR}/test.sh"

cd "${SCRIPT_DIR}"
echo "Synchronizing files..."
echo "Running: ansible-playbook -i ansible/inventory.yml ansible/sync.yml"
if ! ansible-playbook -i ansible/inventory.yml ansible/sync.yml; then
    echo "ERROR: Ansible playbook failed during file synchronization"
    exit 1
fi


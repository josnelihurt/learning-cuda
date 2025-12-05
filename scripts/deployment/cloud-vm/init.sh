#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "${SCRIPT_DIR}/test.sh"

cd "${SCRIPT_DIR}"
echo "Initializing deployment..."
echo "Running: ansible-playbook -i ansible/inventory.yml ansible/init.yml"
if ! ansible-playbook -i ansible/inventory.yml ansible/init.yml; then
    echo "ERROR: Ansible playbook failed during initialization"
    exit 1
fi


#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "${SCRIPT_DIR}/test.sh"

cd "${SCRIPT_DIR}"
echo "Starting services..."
echo "Running: ansible-playbook -i ansible/inventory.yml ansible/start.yml -v"
if ! ansible-playbook -i ansible/inventory.yml ansible/start.yml -v; then
    echo "ERROR: Ansible playbook failed during service startup"
    exit 1
fi


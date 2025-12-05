#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "${SCRIPT_DIR}/test.sh"

cd "${SCRIPT_DIR}"
echo "Synchronizing files..."
ansible-playbook -i ansible/inventory.yml ansible/sync.yml 2>&1


#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "${SCRIPT_DIR}/test.sh"

cd "${SCRIPT_DIR}"
echo "Starting services..."
ansible-playbook -i ansible/inventory.yml ansible/start.yml -v 2>&1


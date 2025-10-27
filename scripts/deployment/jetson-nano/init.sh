#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test.sh"

cd "${SCRIPT_DIR}"
echo "Initializing deployment..."
ansible-playbook -i ansible/inventory.yml ansible/init.yml 2>&1

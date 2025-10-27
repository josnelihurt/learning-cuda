#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test.sh"

cd "${SCRIPT_DIR}"
echo "Cleaning deployment..."
ansible-playbook -i ansible/inventory.yml ansible/clean.yml 2>&1

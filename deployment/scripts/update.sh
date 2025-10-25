#!/bin/bash
# Update CUDA Learning Application on Jetson Nano
# Usage: ./deployment/scripts/update.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

echo -e "${BLUE}Starting CUDA Learning update on Jetson Nano${NC}"
echo "Project root: ${PROJECT_ROOT}"
echo "Deployment dir: ${DEPLOYMENT_DIR}"

# Check if Ansible is installed
if ! command -v ansible-playbook &> /dev/null; then
    echo -e "${RED}Ansible is not installed. Please install it first:${NC}"
    echo "  Ubuntu/Debian: sudo apt install ansible"
    echo "  macOS: brew install ansible"
    echo "  Or: pip install ansible"
    exit 1
fi

# Check if inventory file exists
if [[ ! -f "${DEPLOYMENT_DIR}/ansible/inventory.yml" ]]; then
    echo -e "${RED}Inventory file not found: ${DEPLOYMENT_DIR}/ansible/inventory.yml${NC}"
    exit 1
fi

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection to Jetson Nano...${NC}"
if ! ansible jetson -i "${DEPLOYMENT_DIR}/ansible/inventory.yml" -m ping; then
    echo -e "${RED}SSH connection failed. Please check:${NC}"
    echo "  - SSH key is properly configured"
    echo "  - Jetson Nano is accessible at 192.168.10.213"
    echo "  - User 'jrb' has sudo privileges"
    exit 1
fi

echo -e "${GREEN}SSH connection successful${NC}"

# Run update
echo -e "${YELLOW}Running Ansible playbook for application update...${NC}"
cd "${DEPLOYMENT_DIR}"

ansible-playbook \
    -i ansible/inventory.yml \
    ansible/playbook.yml \
    --tags update \
    --extra-vars "update_mode=true" \
    --ask-become-pass \
    -v

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Update completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Verification steps:${NC}"
    echo "  1. Check application status: ssh jrb@192.168.10.213 'cd /opt/cuda-learning && docker-compose ps'"
    echo "  2. View recent logs: ssh jrb@192.168.10.213 'cd /opt/cuda-learning && docker-compose logs --tail=50'"
    echo "  3. Test application: curl -k https://192.168.10.213"
    echo ""
    echo -e "${BLUE}Available services:${NC}"
    echo "  - Main app: https://192.168.10.213"
    echo "  - Grafana: https://192.168.10.213/grafana"
    echo "  - Jaeger: https://192.168.10.213/jaeger"
    echo "  - Flipt: https://192.168.10.213/flipt"
    echo ""
    echo -e "${YELLOW}Note: Compilation on ARM64 may take 30-60 minutes${NC}"
    echo "   Monitor progress: ssh jrb@192.168.10.213 'cd /opt/cuda-learning && docker-compose logs -f'"
else
    echo -e "${RED}Update failed. Check the output above for errors.${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  1. Check logs: ssh jrb@192.168.10.213 'cd /opt/cuda-learning && docker-compose logs'"
    echo "  2. Rollback: ssh jrb@192.168.10.213 'cd /opt/cuda-learning && docker-compose down && docker-compose up -d'"
    echo "  3. Manual update: ssh jrb@192.168.10.213 'cd /opt/cuda-learning/repo && git pull && docker-compose build && docker-compose up -d'"
    exit 1
fi

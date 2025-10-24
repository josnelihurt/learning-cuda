#!/bin/bash
# Full deployment validation script for CUDA Learning Application
# This script removes the remote directory and deploys from scratch
# Usage: ./deployment/scripts/full-deploy.sh

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

echo -e "${BLUE}Starting full deployment validation${NC}"
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

# Check if playbook exists
if [[ ! -f "${DEPLOYMENT_DIR}/ansible/playbook.yml" ]]; then
    echo -e "${RED}Playbook not found: ${DEPLOYMENT_DIR}/ansible/playbook.yml${NC}"
    exit 1
fi

# Check if sudo password file exists
SUDO_PASSWORD_FILE="${PROJECT_ROOT}/.secrets/sudo_password"
if [[ ! -f "${SUDO_PASSWORD_FILE}" ]]; then
    echo -e "${RED}Sudo password file not found: ${SUDO_PASSWORD_FILE}${NC}"
    echo "Please create the file with the sudo password:"
    echo "  echo 'your_sudo_password' > ${SUDO_PASSWORD_FILE}"
    echo "  chmod 600 ${SUDO_PASSWORD_FILE}"
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

# Remove remote directory
echo -e "${YELLOW}Removing remote directory...${NC}"
ansible jetson -i "${DEPLOYMENT_DIR}/ansible/inventory.yml" \
    -m shell \
    -a "sudo rm -rf /opt/josnelihurt/cuda-learning" \
    --become-password-file "${SUDO_PASSWORD_FILE}" \
    -v

echo -e "${GREEN}Remote directory removed${NC}"

# Run full deployment
echo -e "${YELLOW}Running full Ansible deployment...${NC}"
cd "${DEPLOYMENT_DIR}"

ansible-playbook \
    -i ansible/inventory.yml \
    ansible/playbook.yml \
    --tags setup,application,docker-deploy \
    --become-password-file "${SUDO_PASSWORD_FILE}" \
    -v

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Full deployment completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Deployment Summary:${NC}"
    echo "  - Repository cloned and synced"
    echo "  - Docker images built for ARM64"
    echo "  - Services started with production profile"
    echo "  - All health checks passed"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo "  - Application: http://192.168.10.213:8080"
    echo "  - Traefik Dashboard: http://192.168.10.213:8081"
    echo "  - Jaeger UI: http://192.168.10.213:16686"
    echo "  - Grafana: https://192.168.10.213/grafana"
    echo "  - Flipt: https://192.168.10.213/flipt"
    echo "  - Test Reports: https://192.168.10.213/reports"
    echo ""
    echo -e "${YELLOW}Manual verification commands:${NC}"
    echo "  - Check containers: ssh jrb@192.168.10.213 'cd /opt/josnelihurt/cuda-learning && docker compose ps'"
    echo "  - View logs: ssh jrb@192.168.10.213 'cd /opt/josnelihurt/cuda-learning && docker compose logs -f'"
    echo "  - Stop services: ssh jrb@192.168.10.213 'cd /opt/josnelihurt/cuda-learning && docker compose down'"
else
    echo -e "${RED}Full deployment failed. Check the output above for errors.${NC}"
    exit 1
fi

#!/bin/bash
# Deploy CUDA Learning Application to Jetson Nano
# Usage: ./deployment/scripts/deploy.sh [--production]

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

# Parse command line arguments
PRODUCTION_MODE=false
if [[ "${1:-}" == "--production" ]]; then
    PRODUCTION_MODE=true
fi

echo -e "${BLUE}Starting CUDA Learning deployment to Jetson Nano${NC}"
echo "Project root: ${PROJECT_ROOT}"
echo "Deployment dir: ${DEPLOYMENT_DIR}"
echo "Production mode: ${PRODUCTION_MODE}"

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

# Run deployment
echo -e "${YELLOW}Running Ansible playbook to clone repository...${NC}"
cd "${DEPLOYMENT_DIR}"

# Check if sudo password file exists
SUDO_PASSWORD_FILE="${PROJECT_ROOT}/.secrets/sudo_password"
if [[ ! -f "${SUDO_PASSWORD_FILE}" ]]; then
    echo -e "${RED}Sudo password file not found: ${SUDO_PASSWORD_FILE}${NC}"
    echo "Please create the file with the sudo password:"
    echo "  echo 'your_sudo_password' > ${SUDO_PASSWORD_FILE}"
    echo "  chmod 600 ${SUDO_PASSWORD_FILE}"
    exit 1
fi

# Determine which tags to run based on mode
if [[ "${PRODUCTION_MODE}" == "true" ]]; then
    TAGS="setup,application,docker-deploy"
    echo -e "${YELLOW}Running full production deployment...${NC}"
else
    TAGS="setup,application"
    echo -e "${YELLOW}Running repository setup only...${NC}"
fi

ansible-playbook \
    -i ansible/inventory.yml \
    ansible/playbook.yml \
    --tags "${TAGS}" \
    --become-password-file "${SUDO_PASSWORD_FILE}" \
    -v

if [[ $? -eq 0 ]]; then
    if [[ "${PRODUCTION_MODE}" == "true" ]]; then
        echo -e "${GREEN}Full production deployment completed successfully!${NC}"
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
        echo -e "${GREEN}Repository cloned successfully!${NC}"
        echo ""
        echo -e "${BLUE}Next steps (execute manually on Jetson Nano):${NC}"
        echo "  1. SSH to Jetson: ssh jrb@192.168.10.213"
        echo "  2. Navigate to repo: cd /opt/josnelihurt/cuda-learning"
        echo "  3. Build with ARM64: docker compose -f docker-compose.yml build --pull"
        echo "  4. Start services: docker compose --profile production up -d"
        echo ""
        echo -e "${BLUE}Manual commands:${NC}"
        echo "  - Check status: docker compose ps"
        echo "  - View logs: docker compose logs -f"
        echo "  - Stop services: docker compose down"
        echo ""
        echo -e "${YELLOW}Note: Dockerfile supports multi-platform builds automatically${NC}"
        echo -e "${YELLOW}For full production deployment, run: ./deployment/scripts/deploy.sh --production${NC}"
    fi
else
    echo -e "${RED}Deployment failed. Check the output above for errors.${NC}"
    exit 1
fi

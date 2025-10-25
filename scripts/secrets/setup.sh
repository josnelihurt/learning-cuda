#!/bin/bash
#
# Secrets Setup Utility
#
# This script helps set up the secrets management system
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Secrets Management Setup${NC}"
echo ""

# Create .secrets directory if it doesn't exist
mkdir -p .secrets

# Check if development secrets exist
if [ ! -f ".secrets/development.env" ]; then
    echo -e "${YELLOW}Setting up development secrets...${NC}"
    if [ -f ".secrets/development.env.example" ]; then
        cp .secrets/development.env.example .secrets/development.env
        echo "  Created .secrets/development.env from template"
        echo "  Please edit .secrets/development.env with your development values"
    else
        echo -e "${RED}  Development template not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}  Development secrets already exist${NC}"
fi

# Check if production secrets exist
if [ ! -f ".secrets/production.env" ]; then
    echo -e "${YELLOW}Setting up production secrets...${NC}"
    if [ -f ".secrets/production.env.example" ]; then
        cp .secrets/production.env.example .secrets/production.env
        echo "  Created .secrets/production.env from template"
        echo "  Please edit .secrets/production.env with your production values"
    else
        echo -e "${RED}  Production template not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}  Production secrets already exist${NC}"
fi

# Set proper permissions
chmod 600 .secrets/*.env 2>/dev/null || true
echo "  Set secure permissions on secret files"

echo ""
echo -e "${GREEN}Secrets setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit .secrets/development.env for development"
echo "  2. Edit .secrets/production.env for production"
echo "  3. Run ./scripts/dev/start.sh for development"
echo "  4. Run ./scripts/prod/start.sh for production"
echo ""
echo -e "${YELLOW}Remember: Never commit .env files to version control!${NC}"

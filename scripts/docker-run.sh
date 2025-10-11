#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

./scripts/validate-docker-env.sh

if [ $? -ne 0 ]; then
    exit 1
fi

DETACHED=false
for arg in "$@"; do
    if [[ "$arg" == "-d" ]] || [[ "$arg" == "--detach" ]]; then
        DETACHED=true
    fi
done

if [ "$DETACHED" = true ]; then
    docker-compose up -d --build
    echo ""
    echo "Containers started. Access at https://localhost"
    echo "Traefik dashboard: http://localhost:8081"
    echo ""
    echo "View logs: docker-compose logs -f app"
    echo "Stop: docker-compose down"
else
    docker-compose up --build
fi

#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Starting E2E tests..."
echo "User: $USER_ID:$GROUP_ID"
echo ""

mkdir -p webserver/web/.ignore/test-results
mkdir -p webserver/web/.ignore/playwright-report

echo "Checking services (Flipt + App)..."
if ! curl -s http://localhost:8081/api/v1/health > /dev/null 2>&1; then
    echo "Flipt is not accessible at http://localhost:8081"
    echo "Starting services..."
    ./scripts/start-dev.sh
    
    timeout=30
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8081/api/v1/health > /dev/null 2>&1; then
            echo "Flipt is ready"
            break
        fi
        sleep 1
        timeout=$((timeout - 1))
    done
    if [ $timeout -eq 0 ]; then
        echo "ERROR: Flipt failed to start"
        exit 1
    fi
fi

if ! curl -k -s https://localhost:8443/health > /dev/null 2>&1; then
    echo "ERROR: Service is not accessible at https://localhost:8443"
    echo "Please start the webserver with: ./scripts/start-dev.sh"
    exit 1
fi

echo "Services are running"
echo ""

echo "Building E2E test container..."
docker compose -f docker-compose.dev.yml --profile testing build e2e-tests

echo ""
echo "Running E2E tests..."
docker compose -f docker-compose.dev.yml --profile testing run \
  --rm \
  e2e-tests

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "All E2E tests passed"
else
    echo "E2E tests failed with exit code $EXIT_CODE"
fi

echo ""
echo "Results saved in webserver/web/.ignore/"
echo "  - test-results/e2e-results.json"
echo "  - test-results/e2e-junit.xml"
echo "  - playwright-report/"
echo ""
echo "To view test reports:"
echo "  docker compose -f docker-compose.dev.yml --profile testing up -d e2e-report-viewer"
echo ""
echo "  HTML Report: http://localhost:5051"
echo ""

exit $EXIT_CODE


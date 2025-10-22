#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

TEST_TYPE="${1:-all}"

if [[ "$TEST_TYPE" != "backend" && "$TEST_TYPE" != "e2e" && "$TEST_TYPE" != "all" ]]; then
    echo "Usage: $0 [backend|e2e|all]"
    echo ""
    echo "  backend  - Run only backend BDD tests"
    echo "  e2e      - Run only E2E frontend tests"
    echo "  all      - Run both test suites (default)"
    exit 1
fi

echo "Running tests with Docker..."
echo "Test type: $TEST_TYPE"
echo "User: $USER_ID:$GROUP_ID"
echo ""

mkdir -p integration/tests/acceptance/.ignore/test-results
mkdir -p webserver/web/.ignore/test-results
mkdir -p webserver/web/.ignore/playwright-report

echo "Note: Tests require local services running (Flipt + App)"
echo "Make sure you've run: ./scripts/dev/start.sh"
echo ""

if ! curl -s http://localhost:8081/api/v1/health > /dev/null 2>&1; then
    echo "ERROR: Flipt is not accessible at http://localhost:8081"
    echo "Please start services with: ./scripts/dev/start.sh"
    exit 1
fi

if ! curl -k -s https://localhost:8443/health > /dev/null 2>&1; then
    echo "ERROR: Service is not accessible at https://localhost:8443"
    echo "Please start services with: ./scripts/dev/start.sh"
    exit 1
fi

echo "Services are running"
echo ""

BACKEND_EXIT=0
E2E_EXIT=0

if [[ "$TEST_TYPE" == "backend" || "$TEST_TYPE" == "all" ]]; then
    echo "========================================="
    echo "Running Backend BDD Tests"
    echo "========================================="
    echo ""
    
    docker compose -f docker-compose.dev.yml --profile testing build integration-tests
    
    docker compose -f docker-compose.dev.yml --profile testing run \
      --rm \
      integration-tests
    
    BACKEND_EXIT=$?
    
    echo ""
    if [ $BACKEND_EXIT -eq 0 ]; then
        echo "Backend tests passed"
    else
        echo "Backend tests failed with exit code $BACKEND_EXIT"
    fi
    echo ""
fi

if [[ "$TEST_TYPE" == "e2e" || "$TEST_TYPE" == "all" ]]; then
    echo "========================================="
    echo "Running E2E Frontend Tests"
    echo "========================================="
    echo ""
    
    docker compose -f docker-compose.dev.yml --profile testing build e2e-tests
    
    docker compose -f docker-compose.dev.yml --profile testing run \
      --rm \
      e2e-tests
    
    E2E_EXIT=$?
    
    echo ""
    if [ $E2E_EXIT -eq 0 ]; then
        echo "E2E tests passed"
    else
        echo "E2E tests failed with exit code $E2E_EXIT"
    fi
    echo ""
fi

echo "========================================="
echo "Test Summary"
echo "========================================="

if [[ "$TEST_TYPE" == "backend" || "$TEST_TYPE" == "all" ]]; then
    if [ $BACKEND_EXIT -eq 0 ]; then
        echo "Backend: PASSED"
    else
        echo "Backend: FAILED (exit code $BACKEND_EXIT)"
    fi
fi

if [[ "$TEST_TYPE" == "e2e" || "$TEST_TYPE" == "all" ]]; then
    if [ $E2E_EXIT -eq 0 ]; then
        echo "E2E:     PASSED"
    else
        echo "E2E:     FAILED (exit code $E2E_EXIT)"
    fi
fi

echo ""
echo "========================================="
echo "Test Reports"
echo "========================================="

if [[ "$TEST_TYPE" == "backend" || "$TEST_TYPE" == "all" ]]; then
    echo ""
    echo "Backend Results:"
    echo "  Location: integration/tests/acceptance/.ignore/test-results/"
    echo "  Files: cucumber-report.json, junit-report.xml"
    echo ""
    echo "  View report:"
    echo "    docker compose -f docker-compose.dev.yml --profile testing up -d cucumber-report"
    echo "    HTML Report: http://localhost:5050"
fi

if [[ "$TEST_TYPE" == "e2e" || "$TEST_TYPE" == "all" ]]; then
    echo ""
    echo "E2E Results:"
    echo "  Location: webserver/web/.ignore/"
    echo "  Subdirs: test-results/, playwright-report/"
    echo ""
    echo "  View report:"
    echo "    docker compose -f docker-compose.dev.yml --profile testing up -d e2e-report-viewer"
    echo "    HTML Report: http://localhost:5051"
fi

echo ""

FINAL_EXIT=0
if [ $BACKEND_EXIT -ne 0 ] || [ $E2E_EXIT -ne 0 ]; then
    FINAL_EXIT=1
fi

exit $FINAL_EXIT


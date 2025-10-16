#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Running integration tests with Docker..."
echo "User: $USER_ID:$GROUP_ID"

mkdir -p .ignore/test-results
mkdir -p .ignore/allure-reports

rm -f .ignore/test-results/*.json .ignore/test-results/*.xml 2>/dev/null || true

echo ""
echo "Note: Tests require local services running (Flipt + App)"
echo "Make sure you've run: ./scripts/start-dev.sh"
echo ""

if ! curl -s http://localhost:8081/api/v1/health > /dev/null 2>&1; then
    echo "ERROR: Flipt is not accessible at http://localhost:8081"
    echo "Please start services with: ./scripts/start-dev.sh"
    exit 1
fi

if ! curl -k -s https://localhost:8443/health > /dev/null 2>&1; then
    echo "ERROR: Service is not accessible at https://localhost:8443"
    echo "Please start services with: ./scripts/start-dev.sh"
    exit 1
fi

echo "✅ Services are running"
echo ""

docker compose -f docker-compose.dev.yml --profile testing build integration-tests

docker compose -f docker-compose.dev.yml --profile testing run \
  --rm \
  integration-tests

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Tests failed with exit code $EXIT_CODE"
fi

echo ""
echo "Results saved in integration/tests/acceptance/.ignore/test-results/"
echo "  - cucumber-report.json (Cucumber JSON)"
echo "  - junit-report.xml (JUnit XML)"
echo ""
echo "To view test reports:"
echo "  docker compose -f docker-compose.dev.yml --profile testing up -d cucumber-report"
echo ""
echo "  📊 HTML Report: http://localhost:5050"
echo ""

exit $EXIT_CODE


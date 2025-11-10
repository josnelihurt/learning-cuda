#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

# Helper to read version files without whitespace.
read_version() {
    local path="$1"
    tr -d '[:space:]' < "$PROJECT_ROOT/$path"
}

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
esac

DEFAULT_PROTO_VERSION="$(read_version proto/VERSION)"
DEFAULT_CPP_VERSION="$(read_version cpp_accelerator/VERSION)"
DEFAULT_GOLANG_VERSION="$(read_version webserver/VERSION)"
DEFAULT_INTEGRATION_VERSION="$(read_version integration/VERSION)"

LOCAL_REGISTRY="local/josnelihurt/learning-cuda"
REMOTE_REGISTRY="ghcr.io/josnelihurt/learning-cuda"

REMOTE_GOLANG_IMAGE="${REMOTE_REGISTRY}/intermediate:golang-built-${DEFAULT_GOLANG_VERSION}-${ARCH}"
LOCAL_GOLANG_IMAGE="${LOCAL_REGISTRY}/intermediate:golang-built-${DEFAULT_GOLANG_VERSION}-${ARCH}"

if docker image inspect "$REMOTE_GOLANG_IMAGE" >/dev/null 2>&1; then
    DEFAULT_BASE_REGISTRY="$REMOTE_REGISTRY"
elif docker image inspect "$LOCAL_GOLANG_IMAGE" >/dev/null 2>&1; then
    DEFAULT_BASE_REGISTRY="$LOCAL_REGISTRY"
else
    DEFAULT_BASE_REGISTRY="$REMOTE_REGISTRY"
fi
DEFAULT_BASE_TAG="latest"

export PROTO_VERSION="${PROTO_VERSION:-$DEFAULT_PROTO_VERSION}"
export CPP_VERSION="${CPP_VERSION:-$DEFAULT_CPP_VERSION}"
export GOLANG_VERSION="${GOLANG_VERSION:-$DEFAULT_GOLANG_VERSION}"
export INTEGRATION_VERSION="${INTEGRATION_VERSION:-$DEFAULT_INTEGRATION_VERSION}"
export BASE_REGISTRY="${BASE_REGISTRY:-$DEFAULT_BASE_REGISTRY}"
export BASE_TAG="${BASE_TAG:-$DEFAULT_BASE_TAG}"

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
echo "Docker build context:"
echo "  BASE_REGISTRY:     $BASE_REGISTRY"
echo "  BASE_TAG:          $BASE_TAG"
echo "  PROTO_VERSION:     $PROTO_VERSION"
echo "  CPP_VERSION:       $CPP_VERSION"
echo "  GOLANG_VERSION:    $GOLANG_VERSION"
echo ""

mkdir -p integration/tests/acceptance/.ignore/test-results
mkdir -p .ignore/webserver/web/test-results
mkdir -p .ignore/webserver/web/playwright-report

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
      --user "${USER_ID}:${GROUP_ID}" \
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
      --user "${USER_ID}:${GROUP_ID}" \
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
    echo "  Location: .ignore/webserver/web/"
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


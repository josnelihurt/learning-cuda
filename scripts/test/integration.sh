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
REMOTE_GOLANG_LATEST="${REMOTE_REGISTRY}/intermediate:golang-built-latest-${ARCH}"
LOCAL_GOLANG_LATEST="${LOCAL_REGISTRY}/intermediate:golang-built-latest-${ARCH}"
LOCAL_GOLANG_ALT="local/intermediate:golang-built-latest-${ARCH}"

if docker image inspect "$REMOTE_GOLANG_IMAGE" >/dev/null 2>&1; then
    DEFAULT_BASE_REGISTRY="$REMOTE_REGISTRY"
    DEFAULT_BASE_TAG="latest"
elif docker image inspect "$LOCAL_GOLANG_IMAGE" >/dev/null 2>&1; then
    DEFAULT_BASE_REGISTRY="$LOCAL_REGISTRY"
    DEFAULT_BASE_TAG="latest"
elif docker image inspect "$REMOTE_GOLANG_LATEST" >/dev/null 2>&1; then
    DEFAULT_BASE_REGISTRY="$REMOTE_REGISTRY"
    DEFAULT_BASE_TAG="latest"
elif docker image inspect "$LOCAL_GOLANG_LATEST" >/dev/null 2>&1; then
    DEFAULT_BASE_REGISTRY="$LOCAL_REGISTRY"
    DEFAULT_BASE_TAG="latest"
elif docker image inspect "$LOCAL_GOLANG_ALT" >/dev/null 2>&1; then
    DEFAULT_BASE_REGISTRY="localhost"
    DEFAULT_BASE_TAG="latest"
else
    DEFAULT_BASE_REGISTRY="$REMOTE_REGISTRY"
    DEFAULT_BASE_TAG="latest"
fi

export BASE_REGISTRY="${BASE_REGISTRY:-$DEFAULT_BASE_REGISTRY}"
export BASE_TAG="${BASE_TAG:-$DEFAULT_BASE_TAG}"

# Check if versioned images exist, otherwise use "latest"
check_image_exists() {
    local image="$1"
    docker image inspect "$image" >/dev/null 2>&1
}

# Try different image name formats for local images
check_image_exists_any() {
    local base_name="$1"
    local version="$2"
    local arch="$3"
    
    # Try with full registry path
    check_image_exists "${BASE_REGISTRY}/intermediate:${base_name}-${version}-${arch}" && return 0
    # Try with local/ prefix
    check_image_exists "local/intermediate:${base_name}-${version}-${arch}" && return 0
    # Try with localhost/ prefix
    check_image_exists "localhost/intermediate:${base_name}-${version}-${arch}" && return 0
    return 1
}

PROTO_IMAGE="${BASE_REGISTRY}/intermediate:proto-generated-${DEFAULT_PROTO_VERSION}-${ARCH}"
CPP_IMAGE="${BASE_REGISTRY}/intermediate:cpp-built-${DEFAULT_CPP_VERSION}-${ARCH}"
GOLANG_IMAGE="${BASE_REGISTRY}/intermediate:golang-built-${DEFAULT_GOLANG_VERSION}-${ARCH}"

if check_image_exists "$PROTO_IMAGE"; then
    export PROTO_VERSION="$DEFAULT_PROTO_VERSION"
elif check_image_exists_any "proto-generated" "latest" "$ARCH"; then
    echo "WARNING: Image $PROTO_IMAGE not found, using 'latest' tag"
    export PROTO_VERSION="latest"
    # Update BASE_REGISTRY if needed
    if check_image_exists "local/intermediate:proto-generated-latest-${ARCH}"; then
        export BASE_REGISTRY="localhost"
    fi
else
    echo "ERROR: No proto-generated image found (tried versioned and latest)"
    exit 1
fi

if check_image_exists "$CPP_IMAGE"; then
    export CPP_VERSION="$DEFAULT_CPP_VERSION"
elif check_image_exists_any "cpp-built" "latest" "$ARCH"; then
    echo "WARNING: Image $CPP_IMAGE not found, using 'latest' tag"
    export CPP_VERSION="latest"
    if check_image_exists "local/intermediate:cpp-built-latest-${ARCH}"; then
        export BASE_REGISTRY="localhost"
    fi
else
    echo "ERROR: No cpp-built image found (tried versioned and latest)"
    exit 1
fi

if check_image_exists "$GOLANG_IMAGE"; then
    export GOLANG_VERSION="$DEFAULT_GOLANG_VERSION"
elif check_image_exists_any "golang-built" "latest" "$ARCH"; then
    echo "WARNING: Image $GOLANG_IMAGE not found, using 'latest' tag"
    export GOLANG_VERSION="latest"
    if check_image_exists "local/intermediate:golang-built-latest-${ARCH}"; then
        export BASE_REGISTRY="localhost"
    fi
else
    echo "ERROR: No golang-built image found (tried versioned and latest)"
    exit 1
fi

export INTEGRATION_VERSION="${INTEGRATION_VERSION:-$DEFAULT_INTEGRATION_VERSION}"

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


#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

BROWSER=""
PLAYWRIGHT_OPTS=""
SHOW_HELP=false
WORKERS=""

for arg in "$@"; do
    case $arg in
        --chromium|--chrome|-c)
            BROWSER="chromium"
            ;;
        --firefox|-f)
            BROWSER="firefox"
            ;;
        --webkit|--safari|-w)
            BROWSER="webkit"
            ;;
        --headed)
            PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS --headed"
            ;;
        --debug)
            PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS --debug"
            ;;
        --ui)
            PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS --ui"
            ;;
        --workers=*)
            WORKERS="${arg#*=}"
            ;;
        --help|-h)
            SHOW_HELP=true
            ;;
        *)
            PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS $arg"
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: ./scripts/run-e2e-tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --chromium, -c      Run tests only in Chromium"
    echo "  --firefox, -f       Run tests only in Firefox"
    echo "  --webkit, -w        Run tests only in WebKit/Safari"
    echo "  --workers=N         Number of parallel workers (default: 25)"
    echo "  --headed            Run tests in headed mode (visible browser)"
    echo "  --debug             Run tests in debug mode"
    echo "  --ui                Run tests in UI mode"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Environment:"
    echo "  CUDA_PROCESSOR_PROCESSOR_DEFAULT_LIBRARY=2.0.0"
    echo "  System CPUs: $(nproc) cores detected"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run-e2e-tests.sh --chromium              # Fast: Chromium only, 25 workers"
    echo "  ./scripts/run-e2e-tests.sh --chromium --workers=30 # Ultra fast: 30 workers"
    echo "  ./scripts/run-e2e-tests.sh --headed                # All browsers, visible"
    echo "  ./scripts/run-e2e-tests.sh --chromium --debug      # Debug Chromium tests"
    exit 0
fi

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
export CUDA_PROCESSOR_PROCESSOR_DEFAULT_LIBRARY=2.0.0

if [ -n "$BROWSER" ]; then
    PLAYWRIGHT_OPTS="--project=$BROWSER $PLAYWRIGHT_OPTS"
fi

if [ -n "$WORKERS" ]; then
    export PLAYWRIGHT_WORKERS="$WORKERS"
    PLAYWRIGHT_OPTS="--workers=$WORKERS $PLAYWRIGHT_OPTS"
fi

echo "Starting E2E tests..."
echo "User: $USER_ID:$GROUP_ID"
echo "Processor Library: 2.0.0"
echo "System CPUs: $(nproc) cores"
[ -n "$WORKERS" ] && echo "Workers: $WORKERS" || echo "Workers: 25 (default)"
[ -n "$BROWSER" ] && echo "Browser: $BROWSER" || echo "Browsers: All (chromium, firefox, webkit)"
[ -n "$PLAYWRIGHT_OPTS" ] && echo "Playwright Options: $PLAYWRIGHT_OPTS"
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
echo "Command: npx playwright test $PLAYWRIGHT_OPTS"
echo ""

docker compose -f docker-compose.dev.yml --profile testing run \
  --rm \
  -e PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS --grep-invert=@slow" \
  -e PLAYWRIGHT_WORKERS="${PLAYWRIGHT_WORKERS:-25}" \
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


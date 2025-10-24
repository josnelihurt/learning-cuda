#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Validating test prerequisites..."
REQUIRED_VIDEO="$PROJECT_ROOT/data/test-data/videos/e2e-test.mp4"
if [ ! -f "$REQUIRED_VIDEO" ]; then
    echo "ERROR: Test video not found: $REQUIRED_VIDEO"
    echo "Generate it with: ./scripts/tools/generate-video.sh"
    exit 1
fi

FRAMES_DIR="$PROJECT_ROOT/data/test-data/video-frames/e2e-test"
if [ ! -d "$FRAMES_DIR" ] || [ -z "$(ls -A $FRAMES_DIR 2>/dev/null)" ]; then
    echo "WARNING: Video frames not found, extracting..."
    ./scripts/tools/extract-frames.sh
fi

echo "Prerequisites validated"
echo ""

BROWSER=""
PLAYWRIGHT_OPTS=""
SHOW_HELP=false
WORKERS=""
ENVIRONMENT="dev"

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
        --env=*)
            ENVIRONMENT="${arg#*=}"
            ;;
        --dev)
            ENVIRONMENT="dev"
            ;;
        --prod)
            ENVIRONMENT="prod"
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
    echo "Usage: ./scripts/test/e2e.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --chromium, -c      Run tests only in Chromium"
    echo "  --firefox, -f       Run tests only in Firefox"
    echo "  --webkit, -w        Run tests only in WebKit/Safari"
    echo "  --workers=N         Number of parallel workers (default: 25)"
    echo "  --headed            Run tests in headed mode (visible browser)"
    echo "  --debug             Run tests in debug mode"
    echo "  --ui                Run tests in UI mode"
    echo "  --env=ENV           Set environment (dev|prod, default: dev)"
    echo "  --dev               Run tests in development environment (port 8443)"
    echo "  --prod              Run tests in production environment (port 443)"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Environment:"
    echo "  Development: https://localhost:8443 (default)"
    echo "  Production:  https://localhost:443"
    echo "  CUDA_PROCESSOR_PROCESSOR_DEFAULT_LIBRARY=2.0.0"
    echo "  System CPUs: $(nproc) cores detected"
    echo ""
    echo "Examples:"
    echo "  ./scripts/test/e2e.sh --chromium              # Fast: Chromium only, dev env"
    echo "  ./scripts/test/e2e.sh --prod --chromium       # Production environment"
    echo "  ./scripts/test/e2e.sh --dev --headed          # Development, visible browser"
    echo "  ./scripts/test/e2e.sh --prod --debug          # Production debug mode"
    exit 0
fi

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)
export CUDA_PROCESSOR_PROCESSOR_DEFAULT_LIBRARY=2.0.0

# Set environment variables based on environment
if [ "$ENVIRONMENT" = "prod" ]; then
    export TEST_ENV="production"
    export PLAYWRIGHT_BASE_URL="https://localhost:443"
    echo "Environment: Production (https://localhost:443)"
else
    export TEST_ENV="development"
    export PLAYWRIGHT_BASE_URL="https://localhost:8443"
    echo "Environment: Development (https://localhost:8443)"
fi

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
    if [ "$ENVIRONMENT" = "prod" ]; then
        echo "For production, make sure Docker Compose services are running:"
        echo "  docker compose --profile production up -d"
        exit 1
    else
        echo "Starting development services..."
        ./scripts/dev/start.sh
        
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
fi

# Check application health based on environment
if [ "$ENVIRONMENT" = "prod" ]; then
    if ! curl -k -s https://localhost:443/health > /dev/null 2>&1; then
        echo "ERROR: Production service is not accessible at https://localhost:443"
        echo "Please start production services with: docker compose --profile production up -d"
        exit 1
    fi
else
    if ! curl -k -s https://localhost:8443/health > /dev/null 2>&1; then
        echo "ERROR: Development service is not accessible at https://localhost:8443"
        echo "Please start the webserver with: ./scripts/dev/start.sh"
        exit 1
    fi
fi

echo "Services are running"
echo ""

echo "Building E2E test container..."
docker compose -f docker-compose.dev.yml --profile testing build e2e-tests

echo ""
echo "Running E2E tests..."
echo "Command: npx playwright test $PLAYWRIGHT_OPTS"
echo ""

set +e
docker compose -f docker-compose.dev.yml --profile testing run \
  --rm \
  -e PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS --grep-invert=@slow" \
  -e PLAYWRIGHT_WORKERS="${PLAYWRIGHT_WORKERS:-25}" \
  -e TEST_ENV="$TEST_ENV" \
  -e PLAYWRIGHT_BASE_URL="$PLAYWRIGHT_BASE_URL" \
  e2e-tests

EXIT_CODE=$?
set -e

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "All E2E tests passed"
else
    echo "E2E tests failed with exit code $EXIT_CODE"
fi


docker stop e2e-report-viewer
docker rm e2e-report-viewer

docker compose -f docker-compose.dev.yml --profile testing up -d e2e-report-viewer

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


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
        --staging)
            ENVIRONMENT="staging"
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
    echo "  --env=ENV           Set environment (dev|prod|staging, default: dev)"
    echo "  --dev               Run tests in development environment (port 8443)"
    echo "  --prod              Run tests in production environment (port 443)"
    echo "  --staging           Run tests in staging environment (.localhost)"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Environment:"
    echo "  Development: https://localhost:8443 (default)"
    echo "  Production:  https://localhost:443"
    echo "  Staging:     https://app.localhost"
    echo "  System CPUs: $(nproc) cores detected"
    echo ""
    echo "Examples:"
    echo "  ./scripts/test/e2e.sh --chromium              # Fast: Chromium only, dev env"
    echo "  ./scripts/test/e2e.sh --prod --chromium       # Production environment"
    echo "  ./scripts/test/e2e.sh --staging --chromium    # Staging environment"
    echo "  ./scripts/test/e2e.sh --dev --headed          # Development, visible browser"
    echo "  ./scripts/test/e2e.sh --prod --debug          # Production debug mode"
    exit 0
fi

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

# Set environment variables based on environment
if [ "$ENVIRONMENT" = "prod" ]; then
    export TEST_ENV="production"
    export PLAYWRIGHT_BASE_URL="https://app-cuda-demo.josnelihurt.me"
    echo "Environment: Production (https://app-cuda-demo.josnelihurt.me)"
elif [ "$ENVIRONMENT" = "staging" ]; then
    export TEST_ENV="staging"
    export PLAYWRIGHT_BASE_URL="https://app.localhost"
    echo "Environment: Staging (https://app.localhost)"
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

# Set Flipt port based on environment
if [ "$ENVIRONMENT" = "prod" ]; then
    FLIPT_PORT="8082"
elif [ "$ENVIRONMENT" = "staging" ]; then
    FLIPT_PORT=""
else
    FLIPT_PORT="8081"
fi

echo "Checking services (Flipt + App)..."
if [ "$ENVIRONMENT" = "prod" ]; then
    # In production, check Flipt via Cloudflare Tunnel
    if ! curl -k -s https://flipt-cuda-demo.josnelihurt.me/api/v1/health > /dev/null 2>&1; then
        echo "Flipt is not accessible at https://flipt-cuda-demo.josnelihurt.me"
        echo "For production, make sure Docker Compose services are running:"
        echo "  docker compose --profile cloudflare up -d"
        exit 1
    fi
elif [ "$ENVIRONMENT" = "staging" ]; then
    # In staging, check Flipt via Traefik
    if ! curl -k -s https://flipt.localhost/api/v1/health > /dev/null 2>&1; then
        echo "Flipt is not accessible at https://flipt.localhost"
        echo "For staging, make sure Docker Compose services are running:"
        echo "  ./scripts/deployment/staging_local/start.sh"
        exit 1
    fi
else
    # In development, check Flipt directly
    if ! curl -s http://localhost:$FLIPT_PORT/api/v1/health > /dev/null 2>&1; then
        echo "Flipt is not accessible at http://localhost:$FLIPT_PORT"
        echo "Starting development services..."
        ./scripts/dev/start.sh
        
        timeout=30
        while [ $timeout -gt 0 ]; do
            if curl -s http://localhost:$FLIPT_PORT/api/v1/health > /dev/null 2>&1; then
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
    if ! curl -k -s https://app-cuda-demo.josnelihurt.me/health > /dev/null 2>&1; then
        echo "ERROR: Production service is not accessible at https://app-cuda-demo.josnelihurt.me"
        echo "Please start production services with: docker compose --profile cloudflare up -d"
        exit 1
    fi
elif [ "$ENVIRONMENT" = "staging" ]; then
    if ! curl -k -s https://app.localhost/health > /dev/null 2>&1; then
        echo "ERROR: Staging service is not accessible at https://app.localhost"
        echo "Please start staging services with: ./scripts/deployment/staging_local/start.sh"
        exit 1
    fi
else
    if ! curl -k -s https://localhost:8443/health > /dev/null 2>&1; then
        echo "ERROR: Development service is not accessible at https://localhost:8443"
        echo "Please start development services with: ./scripts/dev/start.sh"
        exit 1
    fi
fi

echo "Services are running"
echo ""

echo "Running E2E tests locally..."
echo "Command: npx playwright test $PLAYWRIGHT_OPTS"
echo ""

# Change to the web directory where Playwright is configured
cd webserver/web

set +e
npx playwright test $PLAYWRIGHT_OPTS
EXIT_CODE=$?
set -e

# Return to project root
cd "$PROJECT_ROOT"

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
echo "  npx playwright show-report"
echo ""

exit $EXIT_CODE
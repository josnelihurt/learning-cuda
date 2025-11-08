#################################################################################
#                          DOCKERFILE ARGUMENTS                                 #
#################################################################################
ARG BASE_REGISTRY=ghcr.io/josnelihurt/learning-cuda
ARG BASE_TAG=latest
ARG TARGETARCH=amd64
ARG PROTO_VERSION=1.0.0
ARG CPP_VERSION=2.1.0
ARG GOLANG_VERSION=1.0.0
ARG NODE_VERSION=20
ARG PLAYWRIGHT_VERSION=v1.56.1
ARG UBUNTU_VARIANT=jammy

#################################################################################
#                    INTERMEDIATE IMAGES REFERENCES                              #
#################################################################################
# These stages reference the intermediate images from GHCR
# The actual compilation happens in separate workflows
#################################################################################

FROM ${BASE_REGISTRY}/intermediate:proto-generated-${PROTO_VERSION}-${TARGETARCH} AS proto-generated-ref
FROM ${BASE_REGISTRY}/intermediate:cpp-built-${CPP_VERSION}-${TARGETARCH} AS cpp-built-ref
FROM ${BASE_REGISTRY}/intermediate:golang-built-${GOLANG_VERSION}-${TARGETARCH} AS golang-built-ref

#################################################################################
#                            FRONTEND BUILDER STAGE                             #
#################################################################################
# Build TypeScript/Vite frontend into static assets
# Step 1: Generate TypeScript protobufs using buf + node plugins
# Step 2: Build frontend with Vite
# Output: Compiled JS/CSS in /build/static and HTML templates
#################################################################################

FROM ${BASE_REGISTRY}/base:proto-tools-${BASE_TAG}-${TARGETARCH} AS proto-gen-builder

FROM node:${NODE_VERSION}-alpine AS frontend-builder

WORKDIR /build

# Install protoc and buf for protobuf generation
RUN apk add --no-cache protobuf-dev

# Copy buf and Go protobuf plugins from proto-tools base image
COPY --from=proto-gen-builder /go/bin/buf /usr/local/bin/buf
COPY --from=proto-gen-builder /go/bin/protoc-gen-go /usr/local/bin/protoc-gen-go
COPY --from=proto-gen-builder /go/bin/protoc-gen-connect-go /usr/local/bin/protoc-gen-connect-go

# Install npm dependencies (includes TypeScript protobuf plugins)
COPY webserver/web/package*.json ./webserver/web/
RUN cd webserver/web && npm ci

# Copy protobuf definitions and config
COPY buf.yaml buf.lock buf.gen.yaml ./
COPY proto/*.proto ./proto/

# Copy webserver source (needed for protobuf generation paths)
COPY webserver/web/ ./webserver/web/

# Generate TypeScript protobufs
RUN buf generate

# Build frontend with Vite (includes generated protobufs)
WORKDIR /build/webserver/web
RUN npm run build


#################################################################################
#                        INTEGRATION TESTS STAGE                                #
#################################################################################
# Run BDD acceptance tests using Godog
# Requires: Go server binary + C++ libraries to run the full stack
# Tests make HTTP/WebSocket requests to the running server
# This stage is optional and only runs when explicitly targeted
#################################################################################

FROM ${BASE_REGISTRY}/base:integration-tests-base-${BASE_TAG}-${TARGETARCH} AS integration-tests-base

FROM integration-tests-base AS integration-tests
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Copy compiled server binary and C++ libraries from intermediate images
COPY --from=golang-built-ref /artifacts/bin/server /workspace/bin/server
COPY --from=cpp-built-ref /artifacts/lib/ /workspace/.ignore/lib/cuda_learning/

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN mkdir -p /workspace/integration/tests/acceptance/.ignore/test-results && \
    mkdir -p /home/testuser/.cache && \
    groupadd -g ${GROUP_ID} testuser || true && \
    useradd -u ${USER_ID} -g testuser -m -s /bin/bash testuser 2>/dev/null || true && \
    chown -R ${USER_ID}:${GROUP_ID} /workspace/integration/tests/acceptance/.ignore /home/testuser /workspace/bin /workspace/.ignore

USER ${USER_ID}:${GROUP_ID}
ENV HOME=/home/testuser
ENV LD_LIBRARY_PATH=/workspace/.ignore/lib/cuda_learning

WORKDIR /workspace/integration/tests/acceptance

CMD ["sh", "-c", "go test . -run TestFeatures -v"]

#################################################################################
#                         E2E FRONTEND TESTS STAGE                              #
#################################################################################
# Run Playwright E2E tests for frontend UI validation
# Requires: Running webserver accessible via network
# Tests execute in Chromium, Firefox, and WebKit browsers
# This stage ONLY installs dependencies - source code mounted via volumes
# This stage is optional and only runs when explicitly targeted
#################################################################################

FROM mcr.microsoft.com/playwright:${PLAYWRIGHT_VERSION}-${UBUNTU_VARIANT} AS e2e-tests

ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /workspace

# Install Node.js dependencies only
COPY webserver/web/package*.json ./webserver/web/
RUN cd webserver/web && npm ci

# Install Playwright browsers
RUN cd webserver/web && npx playwright install chromium firefox webkit

# Create user with same UID/GID as host
RUN groupadd -g ${GROUP_ID} testuser || true && \
    useradd -u ${USER_ID} -g testuser -m -s /bin/bash testuser 2>/dev/null || true && \
    mkdir -p /workspace/webserver/web/.ignore && \
    mkdir -p /home/testuser/.cache && \
    chown -R ${USER_ID}:${GROUP_ID} /workspace /home/testuser

USER ${USER_ID}:${GROUP_ID}
ENV HOME=/home/testuser
ENV NODE_ENV=test
ENV PLAYWRIGHT_BASE_URL=https://localhost:8443

WORKDIR /workspace/webserver/web

ENTRYPOINT ["sh", "-c", "npx playwright test ${PLAYWRIGHT_OPTS}"]

#################################################################################
#                       TEST REPORTS GENERATOR STAGE                            #
#################################################################################
# Generate HTML test reports from integration test results
# Uses cucumber JSON output to create visual HTML report
# Output: /app/reports/ directory with index.html and assets
#################################################################################

FROM node:${NODE_VERSION}-alpine AS test-reports

WORKDIR /app

RUN mkdir -p /app/input /app/reports

COPY --from=integration-tests /workspace/integration/tests/acceptance/.ignore/test-results/ /app/input/

RUN npm install multiple-cucumber-html-reporter && \
    node -e "const report = require('multiple-cucumber-html-reporter'); \
    const fs = require('fs'); \
    const files = fs.readdirSync('/app/input').filter(f => f.endsWith('.json')); \
    if (files.length > 0) { \
      report.generate({ \
        jsonDir: '/app/input', \
        reportPath: '/app/reports', \
        displayDuration: true, \
        displayReportTime: true, \
        pageTitle: 'Integration Tests - BDD Reports', \
        reportName: 'CUDA Learning - Acceptance Tests', \
        metadata: { \
          browser: { name: 'API Tests', version: '1.0' }, \
          platform: { name: 'Docker', version: 'Production' } \
        } \
      }); \
    } else { \
      fs.writeFileSync('/app/reports/index.html', '<html><head><meta charset=\"utf-8\"><title>No Test Results</title></head><body style=\"font-family: sans-serif; padding: 50px; text-align: center;\"><h1>No test results available</h1><p>Run integration tests to generate reports.</p></body></html>'); \
    }"

#################################################################################
#                            RUNTIME STAGE (FINAL)                              #
#################################################################################
# Minimal runtime image with CUDA runtime (no compiler/build tools)
# Combines: Go server + C++ libraries + frontend static files
# The Go server loads C++ .so dynamically based on config
#################################################################################

FROM ${BASE_REGISTRY}/base:runtime-base-${BASE_TAG}-${TARGETARCH} AS runtime-base

FROM runtime-base AS runtime

WORKDIR /app

# Copy compiled artifacts from intermediate images
COPY --from=golang-built-ref /artifacts/bin/server /app/server
COPY --from=cpp-built-ref /artifacts/lib/ /app/lib/
COPY --from=frontend-builder /build/webserver/web/static/ /app/web/static/
COPY --from=frontend-builder /build/webserver/web/templates/ /app/web/templates/

# Copy runtime data and configuration
COPY data/ /app/data/
COPY config/config.yaml /app/config/config.yaml

# Create production configuration
COPY config/config.production.yaml /app/config/config.production.yaml

# Update shared library cache for dynamic loading
RUN ldconfig

EXPOSE 8080 8443

# Set library path for dynamic loading
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

CMD ["/app/server", "-config=/app/config/config.yaml"]

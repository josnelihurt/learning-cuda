#################################################################################
#                          PROTOBUF GENERATOR STAGE                             #
#################################################################################
# Generate Go and C++ code from protobuf definitions using buf
# This must run first as both C++ and Go builds depend on generated code
# Output: proto/gen/ directory with generated .pb.go and .pb.cc files
#################################################################################

FROM golang:1.23-alpine AS proto-gen-builder
RUN go install github.com/bufbuild/buf/cmd/buf@v1.47.2 && \
    go install connectrpc.com/connect/cmd/protoc-gen-connect-go@v1.17.0 && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.35.2

FROM alpine:3.19 AS proto-generator
RUN apk add --update --no-cache \
    ca-certificates \
    git \
    protoc \
    protobuf-dev \
    && rm -rf /var/cache/apk/*

COPY --from=proto-gen-builder /go/bin/buf /usr/local/bin/buf
COPY --from=proto-gen-builder /go/bin/protoc-gen-go /usr/local/bin/protoc-gen-go
COPY --from=proto-gen-builder /go/bin/protoc-gen-connect-go /usr/local/bin/protoc-gen-connect-go

WORKDIR /workspace
ENV XDG_CACHE_HOME=/workspace/.cache

COPY buf.yaml buf.lock buf.gen.backend.yaml ./
COPY proto/*.proto ./proto/

RUN buf generate --template buf.gen.backend.yaml

#################################################################################
#                            FRONTEND BUILDER STAGE                             #
#################################################################################
# Build TypeScript/Vite frontend into static assets
# Step 1: Generate TypeScript protobufs using buf + node plugins
# Step 2: Build frontend with Vite
# Output: Compiled JS/CSS in /build/static and HTML templates
#################################################################################

FROM node:20-alpine AS frontend-builder

WORKDIR /build

# Install protoc and buf for protobuf generation
RUN apk add --no-cache protobuf-dev

# Copy buf and Go protobuf plugins from proto-gen-builder
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
#                         C++ LIBRARIES BUILDER STAGE                           #
#################################################################################
# Compile CUDA C++ shared libraries (.so) using Bazel + NVIDIA compiler
# Requires: CUDA toolkit, Bazel, protobuf definitions
# Output: libcuda_processor_v{VERSION}.so (VERSION from cpp_accelerator/VERSION)
#################################################################################

FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS cpp-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV BAZEL_VERSION=7.0.2

WORKDIR /build

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    pkg-config \
    zip \
    unzip \
    python3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download Bazel for current platform (using Docker buildx variables)
ARG TARGETARCH
RUN BAZEL_ARCH=$([ "$TARGETARCH" = "amd64" ] && echo "linux-amd64" || echo "linux-arm64") && \
    echo "Downloading Bazel for architecture: $BAZEL_ARCH (TARGETARCH=$TARGETARCH)" && \
    wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-${BAZEL_ARCH} \
    && chmod +x /usr/local/bin/bazel

# Copy only files needed for C++ compilation
# This includes Bazel workspace files, C++ source, and generated protobuf code
COPY MODULE.bazel MODULE.bazel.lock WORKSPACE.bazel BUILD.bazel ./
COPY third_party/ ./third_party/
COPY cpp_accelerator/ ./cpp_accelerator/
COPY proto/*.proto ./proto/
COPY proto/BUILD ./proto/BUILD
COPY buf.yaml buf.lock buf.gen.yaml ./

# Copy generated protobuf code from proto-generator stage
COPY --from=proto-generator /workspace/proto/gen/ ./proto/gen/

RUN bazel build //cpp_accelerator/ports/shared_lib:libcuda_processor.so

RUN VERSION=$(cat cpp_accelerator/VERSION) && \
    mkdir -p /artifacts/lib && \
    cp -L bazel-bin/cpp_accelerator/ports/shared_lib/libcuda_processor.so /artifacts/lib/libcuda_processor_v${VERSION}.so

#################################################################################
#                          GO WEBSERVER BUILDER STAGE                           #
#################################################################################
# Compile Go webserver using standard Go toolchain (no Bazel)
# The server dynamically loads C++ libraries at runtime via dlopen
# Output: /artifacts/bin/server executable
#################################################################################

FROM golang:1.24-bookworm AS go-builder

RUN apt-get update && apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavdevice-dev \
    pkg-config \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY webserver/ ./webserver/

# Copy generated protobuf code from proto-generator stage
COPY --from=proto-generator /workspace/proto/gen/ ./proto/gen/

WORKDIR /build/webserver

# Enable CGO for real loader compilation
ENV CGO_ENABLED=1

# Skip stub creation - use real loader with CGO
# The real loader is in webserver/pkg/infrastructure/processor/loader/
# and will be compiled with CGO enabled

RUN make build

RUN mkdir -p /artifacts/bin && \
    cp ../bin/server /artifacts/bin/

#################################################################################
#                        INTEGRATION TESTS STAGE                                #
#################################################################################
# Run BDD acceptance tests using Godog
# Requires: Go server binary + C++ libraries to run the full stack
# Tests make HTTP/WebSocket requests to the running server
# This stage is optional and only runs when explicitly targeted
#################################################################################

FROM ubuntu:24.04 AS integration-tests

ENV DEBIAN_FRONTEND=noninteractive
ENV GO_VERSION=1.24.0

WORKDIR /workspace

# Install Go and runtime dependencies
ARG TARGETARCH
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && GO_ARCH=$([ "$TARGETARCH" = "amd64" ] && echo "linux-amd64" || echo "linux-arm64") && \
    echo "Downloading Go for architecture: $GO_ARCH (TARGETARCH=$TARGETARCH)" && \
    wget https://go.dev/dl/go${GO_VERSION}.${GO_ARCH}.tar.gz \
    && tar -C /usr/local -xzf go${GO_VERSION}.${GO_ARCH}.tar.gz \
    && rm go${GO_VERSION}.${GO_ARCH}.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV GO111MODULE=on

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Copy compiled server binary and C++ libraries from builder stages
COPY --from=go-builder /artifacts/bin/server /workspace/bin/server
COPY --from=cpp-builder /artifacts/lib/ /workspace/.ignore/lib/cuda_learning/

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

FROM mcr.microsoft.com/playwright:v1.56.1-jammy AS e2e-tests

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

FROM node:20-alpine AS test-reports

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

FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies including FFmpeg libraries
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavdevice-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled artifacts from builder stages
COPY --from=go-builder /artifacts/bin/server /app/server
COPY --from=cpp-builder /artifacts/lib/ /app/lib/
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

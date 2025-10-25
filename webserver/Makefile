.PHONY: build test run clean install-deps

BINARY_NAME=server
BUILD_DIR=../bin
GO_FILES=$(shell find . -name '*.go' -type f)

build:
	@echo "Building Go webserver..."
	@mkdir -p $(BUILD_DIR)
	@COMMIT_HASH=$$(git rev-parse --short HEAD 2>/dev/null || echo "dev"); \
	BUILD_TIME=$$(date -u +"%Y-%m-%dT%H:%M:%SZ"); \
	VERSION=$$(cat ../cpp_accelerator/VERSION 2>/dev/null || echo "1.0.0"); \
	BRANCH=$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main"); \
	cd cmd/server && CGO_ENABLED=1 go build \
		-ldflags "-X github.com/jrb/cuda-learning/webserver/pkg/infrastructure/build.version=$$VERSION \
		           -X github.com/jrb/cuda-learning/webserver/pkg/infrastructure/build.branch=$$BRANCH \
		           -X github.com/jrb/cuda-learning/webserver/pkg/infrastructure/build.buildTime=$$BUILD_TIME \
		           -X github.com/jrb/cuda-learning/webserver/pkg/infrastructure/build.commitHash=$$COMMIT_HASH" \
		-o ../../$(BUILD_DIR)/$(BINARY_NAME)
	@echo "✓ Build complete: $(BUILD_DIR)/$(BINARY_NAME)"

test:
	@echo "Running tests..."
	go test ./... -v -cover

test-short:
	@echo "Running short tests..."
	go test ./... -short

run: build
	@echo "Starting webserver..."
	$(BUILD_DIR)/$(BINARY_NAME) -webroot=web

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)/$(BINARY_NAME)
	go clean

install-deps:
	@echo "Installing Go dependencies..."
	go mod download
	go mod tidy

lint:
	@echo "Running linter..."
	go vet ./...
	go fmt ./...

dev:
	@echo "Running in development mode..."
	go run cmd/server/main.go -webroot=web


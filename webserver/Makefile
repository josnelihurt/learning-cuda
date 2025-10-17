.PHONY: build test run clean install-deps

BINARY_NAME=server
BUILD_DIR=../bin
GO_FILES=$(shell find . -name '*.go' -type f)

build:
	@echo "Building Go webserver..."
	@mkdir -p $(BUILD_DIR)
	cd cmd/server && go build -o ../../$(BUILD_DIR)/$(BINARY_NAME)
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


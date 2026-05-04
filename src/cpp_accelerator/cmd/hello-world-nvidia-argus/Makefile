CXX ?= g++
CXXFLAGS := -std=c++23 -Wall -Wextra -O2
CXXFLAGS += $(shell pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0)
CXXFLAGS += -I/usr/src/jetson_multimedia_api/include
LDFLAGS := $(shell pkg-config --libs gstreamer-1.0 gstreamer-app-1.0)
LDFLAGS += -L/usr/lib/aarch64-linux-gnu/nvidia -lnvbufsurface -Wl,-rpath,/usr/lib/aarch64-linux-gnu/nvidia

TARGET := hello-world-nvidia-argus
SRC := main.cpp

.PHONY: build clean run

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

CROSS_IMAGE := hello-world-nvidia-argus-cross
JETSON_SSH  := ssh -i $(CURDIR)/../../../../.ignore/jetson.key -o StrictHostKeyChecking=yes jrb@jetson-nano.local.josnelihurt.me
JETSON_REMOTE := /tmp/hello-world-nvidia-argus

.PHONY: cross-build cross-deploy cross-run jetson-build jetson-deploy jetson-run

cross-build:
	docker buildx build --platform linux/arm64 -t $(CROSS_IMAGE) -f Dockerfile.cross --load .
	@# Extract binary from image to local filesystem
	docker create --name extract-$(CROSS_IMAGE) $(CROSS_IMAGE) /bin/true
	docker cp extract-$(CROSS_IMAGE):/build/hello-world-nvidia-argus ./hello-world-nvidia-argus-arm64
	docker rm extract-$(CROSS_IMAGE)
	@echo "Binary extracted: hello-world-nvidia-argus-arm64"
	@file hello-world-nvidia-argus-arm64

cross-deploy: cross-build
	$(JETSON_SSH) "mkdir -p $(JETSON_REMOTE)"
	scp -i $(CURDIR)/../../../../.ignore/jetson.key -o StrictHostKeyChecking=yes \
		hello-world-nvidia-argus-arm64 \
		jrb@jetson-nano.local.josnelihurt.me:$(JETSON_REMOTE)/hello-world-nvidia-argus

cross-run: cross-deploy
	$(JETSON_SSH) "chmod +x $(JETSON_REMOTE)/hello-world-nvidia-argus && $(JETSON_REMOTE)/hello-world-nvidia-argus --detect-only"

# Build natively on the Jetson (uses real JetPack headers for NvBufSurface).
jetson-build:
	$(JETSON_SSH) "mkdir -p $(JETSON_REMOTE)"
	scp -i $(CURDIR)/../../../../.ignore/jetson.key -o StrictHostKeyChecking=yes \
		main.cpp Makefile \
		jrb@jetson-nano.local.josnelihurt.me:$(JETSON_REMOTE)/
	$(JETSON_SSH) "cd $(JETSON_REMOTE) && make build"

jetson-deploy: jetson-build

jetson-run: jetson-build
	$(JETSON_SSH) "cd $(JETSON_REMOTE) && ./hello-world-nvidia-argus --max-sensor 0 --frames 3 --width 1280 --height 720 --fps 15"

run: $(TARGET)
	./$(TARGET)

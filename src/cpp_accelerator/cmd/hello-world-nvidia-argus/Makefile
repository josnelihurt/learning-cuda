CXX ?= g++
CXXFLAGS := -std=c++23 -Wall -Wextra -O2
CXXFLAGS += $(shell pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0)
LDFLAGS := $(shell pkg-config --libs gstreamer-1.0 gstreamer-app-1.0)

TARGET := hello-world-nvidia-argus
SRC := main.cpp

.PHONY: build clean run

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

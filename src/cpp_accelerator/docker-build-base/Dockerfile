ARG CUDA_VERSION=12.5.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS bazel-base

ARG TARGETARCH=amd64
ARG BAZEL_VERSION=8.4.2
ARG BAZELISK_VERSION=v1.19.0

ENV DEBIAN_FRONTEND=noninteractive
ENV USE_BAZEL_VERSION=${BAZEL_VERSION}

WORKDIR /build

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    pkg-config \
    zip \
    unzip \
    python3 \
    python3-pip \
    ffmpeg \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime for YOLO inference
ARG ONNXRUNTIME_VERSION=1.19.2
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz && \
    cp onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so /usr/local/lib/ && \
    cp -r onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}/include/* /usr/local/include/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}*

RUN BAZEL_ARCH=$([ "$TARGETARCH" = "amd64" ] && echo "linux-amd64" || echo "linux-arm64") && \
    echo "Downloading Bazelisk ${BAZELISK_VERSION} for architecture: $BAZEL_ARCH (TARGETARCH=$TARGETARCH)" && \
    wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-${BAZEL_ARCH} \
    && chmod +x /usr/local/bin/bazel


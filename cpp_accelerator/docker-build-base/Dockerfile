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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN BAZEL_ARCH=$([ "$TARGETARCH" = "amd64" ] && echo "linux-amd64" || echo "linux-arm64") && \
    echo "Downloading Bazelisk ${BAZELISK_VERSION} for architecture: $BAZEL_ARCH (TARGETARCH=$TARGETARCH)" && \
    wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-${BAZEL_ARCH} \
    && chmod +x /usr/local/bin/bazel


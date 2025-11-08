ARG GO_VERSION=1.24
ARG DEBIAN_VARIANT=bookworm

FROM golang:${GO_VERSION}-${DEBIAN_VARIANT} AS go-builder-base

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

ENV CGO_ENABLED=1


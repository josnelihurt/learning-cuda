ARG CUDA_VERSION=12.5.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --allow-releaseinfo-change || apt-get update || true && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavdevice-dev \
    || (echo "Retrying with GPG fix..." && \
        apt-get install -y --no-install-recommends --allow-unauthenticated \
        ca-certificates \
        curl \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libavfilter-dev \
        libavdevice-dev) && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app


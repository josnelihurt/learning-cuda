ARG CUDA_VERSION=12.5.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update --allow-releaseinfo-change || apt-get update || true && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly \
        libavcodec60 \
        libavutil58 \
        libswscale7 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# TensorRT runtime libraries. Required for YOLO inference with the TRT-enabled
# binary (built with --config=trt). Baked into the base image so the apt-get
# install runs once per cuda-runtime version bump instead of on every
# cpp-accelerator rebuild.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libnvinfer10 \
        libnvonnxparsers10 \
    && rm -rf /var/lib/apt/lists/*

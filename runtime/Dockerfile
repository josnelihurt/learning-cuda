ARG DEBIAN_VERSION=bookworm-slim

FROM debian:${DEBIAN_VERSION} AS runtime-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

